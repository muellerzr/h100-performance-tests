# Training script based on TRL, but using just accelerate
# original script: https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py
import inspect
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
# from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, get_linear_schedule_with_warmup
from transformers.utils import find_labels
import time

model_name = "trl-internal-testing/dummy-GPT2-correct-vocab"
dataset_name = "timdettmers/openassistant-guanaco"
dataset_text_field = "text"
learning_rate = 1.41e-5
batch_size = 8
seq_length = 512
gradient_accumulation_steps = 16
peft_lora_r = 64
peft_lora_alpha = 16
num_training_steps=500


quantization_config = BitsAndBytesConfig(
    load_in_8bit=True
)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=quantization_config,
    device_map = {"":0},
    torch_dtype = torch.bfloat16
)
signature = inspect.signature(model.forward)
signature_columns = list(signature.parameters.keys())
signature_columns += list(set(["label", "label_ids"] + find_labels(model.__class__)))


def get_dataloaders(accelerator:Accelerator, batch_size:int = 8):
    dataset = load_dataset(dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ignored_columns = list(set(dataset.column_names) - set(signature_columns))
    dataset = dataset.remove_columns(ignored_columns)

    def collate_fn(examples):
        if accelerator.mixed_precision == "fp8":
            pad_to_multiple_of = 16
        elif accelerator.mixed_precision != "no":
            pad_to_multiple_of = 8
        else:
            pad_to_multiple_of = None
        return tokenizer.pad(
            examples,
            padding="longest",
            pad_to_multiple_of=pad_to_multiple_of,
            return_tensors="pt"
        )

    dataloader_params = {
        "batch_size": batch_size, 
        "collate_fn": collate_fn,
        "drop_last": True,
    }
    train_dataloader = DataLoader(dataset["train"], **dataloader_params)
    eval_dataloader = DataLoader(dataset["validation"], **dataloader_params)
    return train_dataloader, eval_dataloader

accelerator = Accelerator(log_with="wandb", gradient_accumulation_steps=gradient_accumulation_steps)
train_dataloader, eval_dataloader = get_dataloaders(accelerator, batch_size)

optimizer = AdamW(params = model.parameters(), lr=learning_rate)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)

model, optimizer, train_dataloader, eval_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, eval_dataloader, lr_scheduler
)

accelerator.init_trackers("falcon")

model.train()
completed_steps = 0
total_loss = 0
start_time = time.time()
for step, batch in enumerate(train_dataloader):
    with accelerator.accumulate(model):
        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.detach().float()
        accelerator.backward(loss)
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    if accelerator.sync_gradients:
        completed_steps += 1
    
    end_time = time.time()
    total_time = end_time - start_time 
    accelerator.log({"batch_time": total_time})
    start_time = end_time

    if completed_steps >= num_training_steps:
        break

accelerator.end_training()
    


