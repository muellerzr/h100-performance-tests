# Training script based on TRL, but using just accelerate
# original script: https://github.com/huggingface/trl/blob/main/examples/scripts/sft_trainer.py
import torch
from accelerate import Accelerator
from datasets import load_dataset
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer, get_linear_schedule_with_warmup, AutoConfig
from transformers import DataCollatorForLanguageModeling
import time
import sys

from te_llama import TELlamaForCausalLM

from accelerate.utils.dataclasses import FP8RecipeKwargs

assert len(sys.argv) == 5, "CMD ... <precision> <batch_size> <HF_or_TE> <training iters>"
assert sys.argv[1] in ("bf16", "fp8"), "provide correct mixed precision type!"

mixed_precision = sys.argv[1]
model_name = "meta-llama/Llama-2-70b-hf"
dataset_name = "timdettmers/openassistant-guanaco"
dataset_text_field = "text"
learning_rate = 1.41e-5
batch_size = int(sys.argv[2])
max_seq_length = 256
gradient_accumulation_steps = 1
num_training_steps=int(sys.argv[4])

config = AutoConfig.from_pretrained(model_name)

with torch.device("meta"):
    if sys.argv[3] == 'HF':
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
        )
    else:
        model = TELlamaForCausalLM.from_pretrained_local(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
        )

# Load model weights in efficiently
folder = "models/models--meta-llama--Llama-2-7b-hf/snapshots/8cca527612d856d7d32bd94f8103728d614eb852"
paths = [
    f"{folder}/pytorch_model-00001-of-00002.bin",
    f"{folder}/pytorch_model-00002-of-00002.bin",
]
checkpoints = [torch.load(path, mmap=True) for path in paths]
for checkpoint in checkpoints:
    model.load_state_dict(checkpoint, strict=False, assign=True)

model.config.use_cache=False

def get_dataloaders(accelerator:Accelerator, batch_size:int = 8):
    dataset = load_dataset(dataset_name, split="train")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if getattr(tokenizer, "pad_token", None) is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize(element):
        outputs = tokenizer(
            element["text"],
            truncation=True,
            padding=False,
            max_length=max_seq_length,
            return_overflowing_tokens=False,
            return_length=False
        )
        return {"input_ids": outputs["input_ids"], "attention_mask": outputs["attention_mask"]}

    with accelerator.main_process_first():
        dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names
        )

    pad_to_multiple_of = 16
    if accelerator.mixed_precision == "fp8":
        pad_to_multiple_of = 16
    elif accelerator.mixed_precision != "no":
        pad_to_multiple_of = 8


    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=pad_to_multiple_of,
    )

    dataloader_params = {
        "batch_size": batch_size,
        "collate_fn": data_collator,
        "drop_last": True,
    }
    train_dataloader = DataLoader(dataset, **dataloader_params)
    return train_dataloader

fp8_kwarg_handler = [FP8RecipeKwargs(backend="te")] if mixed_precision == "fp8" else None
accelerator = Accelerator(
        log_with="wandb", gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        kwargs_handlers=fp8_kwarg_handler
    )

accelerator.print(f'State: {accelerator.state}')
train_dataloader = get_dataloaders(accelerator, batch_size)

optimizer = AdamW(params = model.parameters(), lr=learning_rate, fused=True)

lr_scheduler = get_linear_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=100,
    num_training_steps=num_training_steps,
)

model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
    model, optimizer, train_dataloader, lr_scheduler
)

accelerator.init_trackers("fp8-llama", config={
    "model_name": model_name,
    "dataset_name": dataset_name,
    "batch_size": batch_size,
    "accelerator_state": accelerator.state,
    "mixed_precision": accelerator.mixed_precision,
},
init_kwargs={"wandb": {"name": f'{accelerator.mixed_precision}_bs_{batch_size}_{accelerator.num_processes}_gpus'}})

model.train()
completed_steps = 0
total_loss = 0
optimizer.zero_grad()
# 100 just to get the full time in
for _ in range(100):
    if completed_steps >= num_training_steps:
        break
    for step, batch in enumerate(train_dataloader):
        start_time = time.time()
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
        accelerator.log({"batch_time": total_time, "input_ids": batch["input_ids"].cpu().numpy(), "attention_mask": batch["attention_mask"].cpu().numpy()})
        start_time = end_time

        if completed_steps >= num_training_steps:
            break

accelerator.end_training()



