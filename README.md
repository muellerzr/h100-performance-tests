# h100 testing scripts

## To run:

1. Clone this repository
2. Install requirements `pip install -r h100-stuff/requirements.txt` (and ensure you have `git-lfs` installed. [See here for directions](https://askubuntu.com/questions/799341/how-to-install-git-lfs-on-ubuntu-16-04))
3. `huggingface-hub login`, and pass in your [Hugging Face API token](http://hf.co/settings/token)
4. `wandb login` to track with wandb
5. `mkdir model` to create a model directory. **Note: Should be done one directory outside this directory**
6. `accelerate launch --config_file h100-stuff/fp8.yml h100-stuff/codeparrot/scripts/codeparrot_training.py --train_batch_size 32 --eval_batch_size 32 --max_train_steps 100 --save_dir model/`

## To run a variety of setups:

### fp8 or bf16 single node

Big model: 

```bash
accelerate launch --config_file h100-stuff/fp8.yml h100-stuff/codeparrot/scripts/codeparrot_training.py --train_batch_size 32 --eval_batch_size 32 --max_train_steps 100 --save_dir model/
```

```bash
accelerate launch --config_file h100-stuff/bf16.yml h100-stuff/codeparrot/scripts/codeparrot_training.py --train_batch_size 32 --eval_batch_size 32 --max_train_steps 100 --save_dir model/
```

Small model:

```bash
accelerate launch --config_file h100-stuff/bf16.yml h100-stuff/codeparrot/scripts/codeparrot_training.py --max_train_steps 100 --save_dir model/ --model_ckpt codeparrot/codeparrot-small --train_batch_size 64 --valid_batch_size 64
```

```bash
accelerate launch --config_file h100-stuff/fp8.yml h100-stuff/codeparrot/scripts/codeparrot_training.py --max_train_steps 100 --save_dir model/ --model_ckpt codeparrot/codeparrot-small --train_batch_size 64 --valid_batch_size 64
```

## fp8 or bf16 on multi-node

### Note: Currently has issues ###

Change each yml to be (maintaining the `mixed_precision` already stored there):

```diff
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
-distributed_type: 'NO'
+distributed_type: 'MULTI_GPU'
downcast_bf16: 'no'
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
mixed_precision: 'bf16'
num_machines: 1
-num_processes: 1
+num_processes: 8
use_cpu: false
```
