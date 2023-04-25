import logging
import os
import time
from argparse import Namespace
from pathlib import Path

import datasets
import torch
from accelerate import Accelerator
# from arguments import TrainingArguments
from datasets import load_dataset
from huggingface_hub import Repository
from torch.optim import AdamW
from torch.utils.data import IterableDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.datapipes.iter.combinatorics import ShufflerIterDataPipe

import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser, get_scheduler, set_seed


class ConstantLengthDataset(IterableDataset):
    """
    Iterable dataset that returns constant length chunks of tokens from stream of text files.
        Args:
            tokenizer (Tokenizer): The processor used for proccessing the data.
            dataset (dataset.Dataset): Dataset with text files.
            infinite (bool): If True the iterator is reset after dataset reaches end else stops.
            seq_length (int): Length of token sequences to return.
            num_of_sequences (int): Number of token sequences to keep in buffer.
            chars_per_token (int): Number of characters per token used to estimate number of tokens in text buffer.
            tokenized (bool): If true we use a pretokenized dataset.
    """

    def __init__(
        self,
        tokenizer,
        dataset,
        infinite=False,
        seq_length=1024,
        num_of_sequences=1024,
        chars_per_token=3.6,
        tokenized=False,
    ):
        self.tokenizer = tokenizer
        self.concat_token_id = tokenizer.bos_token_id
        self.dataset = dataset
        self.seq_length = seq_length
        self.epoch = 0
        self.infinite = infinite
        self.current_size = 0
        self.tokenized = tokenized

        if self.tokenized:
            self.max_buffer_size = seq_length * num_of_sequences
            self.content_field = "input_ids"
        else:
            self.max_buffer_size = seq_length * chars_per_token * num_of_sequences
            self.content_field = "content"

    def __iter__(self):
        iterator = iter(self.dataset)
        more_examples = True
        while more_examples:
            buffer, buffer_len = [], 0
            while True:
                if buffer_len >= self.max_buffer_size:
                    break
                try:
                    buffer.append(next(iterator)[self.content_field])
                    buffer_len += len(buffer[-1])
                except StopIteration:
                    if self.infinite:
                        iterator = iter(self.dataset)
                        self.epoch += 1
                        logger.info(f"Dataset epoch: {self.epoch}")
                    else:
                        more_examples = False
                        break
            if self.tokenized:
                tokenized_inputs = buffer
            else:
                tokenized_inputs = self.tokenizer(buffer, truncation=False)["input_ids"]
            all_token_ids = []
            for tokenized_input in tokenized_inputs:
                all_token_ids.extend(tokenized_input + [self.concat_token_id])
            for i in range(0, len(all_token_ids), self.seq_length):
                input_ids = all_token_ids[i : i + self.seq_length]
                if len(input_ids) == self.seq_length:
                    self.current_size += 1
                    yield torch.tensor(input_ids)

    def shuffle(self, buffer_size=1000):
        return ShufflerIterDataPipe(self, buffer_size=buffer_size)


def setup_logging(args=None):
    logger = logging.getLogger(__name__)
    log_dir = Path("log/")
    log_dir.mkdir(exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(log_dir / "test.txt"), logging.StreamHandler()],
    )
    if accelerator.is_main_process:  # we only want to setup logging once
        logger.setLevel(logging.INFO)
        datasets.utils.logging.set_verbosity_info()
        transformers.utils.logging.set_verbosity_info()
    else:
        run_name = ""
        logger.setLevel(logging.ERROR)
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()
    return logger, run_name


def create_dataloaders(args):
    ds_kwargs = {"streaming": True}
    train_data = load_dataset("codeparrot/codeparrot-clean-train", split="train", **ds_kwargs)
    train_data = train_data.shuffle(buffer_size=10000, seed=42)
    valid_data = load_dataset("codeparrot/codeparrot-clean-valid", split="train", **ds_kwargs)
    train_dataset = ConstantLengthDataset(
        tokenizer, train_data, infinite=True, seq_length=1024, tokenized=False
    )
    valid_dataset = ConstantLengthDataset(
        tokenizer, valid_data, infinite=False, seq_length=1024, tokenized=False
    )
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    eval_dataloader = DataLoader(valid_dataset, batch_size=2)
    return train_dataloader, eval_dataloader

# Settings
# parser = HfArgumentParser(TrainingArguments)
# args = parser.parse_args()

# Accelerator
accelerator = Accelerator()
acc_state = {str(k): str(v) for k, v in accelerator.state.__dict__.items()}

# args = Namespace(**vars(args), **acc_state)
# samples_per_step = accelerator.state.num_processes * args.train_batch_size
# set_seed(args.seed)


# Logging
logger, run_name = setup_logging()
logger.info(f"Created Accelerator: {accelerator.state}")

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("codeparrot/codeparrot-small")
logger.info(f"Loaded model")

# Load dataset and dataloader
train_dataloader, eval_dataloader = create_dataloaders("codeparrot/codeparrot-small")
logger.info(f"Created dataloaders")


train_dataloader, eval_dataloader = accelerator.prepare(
    train_dataloader, eval_dataloader
)

logger.info(f'Started iterating...')
for batch in train_dataloader:
    print(batch)