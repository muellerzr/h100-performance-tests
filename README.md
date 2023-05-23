# h100 testing scripts

## To run:

1. Clone this repository
2. Install requirements `pip install -r h100-stuff/requirements.txt` (and ensure you have `git-lfs` installed. [See here for directions](https://askubuntu.com/questions/799341/how-to-install-git-lfs-on-ubuntu-16-04))
3. `huggingface-hub login`, and pass in your [Hugging Face API token](http://hf.co/settings/token)
4. `wandb login` to track with wandb
5. `mkdir model` to create a model directory. **Note: Should be done one directory outside this directory**
6. `accelerate launch --config_file h100-stuff/fp8.yml h1000-stuff/run_summarization_no_trainer.py --model_name_or_path t5-11b --dataset_name cnn_dailymail --dataset_config "3.0.0" --source_prefix "summarize: " --output_dir tst-summarization --per_device_train_batch_size=4 --per_device_eval_batch_size=4  --with_tracking --report_to "wandb" --max_train_steps 100`


## fp8 or bf16 on multi-node

Change each yml to be (maintaining the `mixed_precision` already stored there):

And pass it to `accelerate launch` under the `--config_file` param

```diff
compute_environment: LOCAL_MACHINE
deepspeed_config: {}
distributed_type: 'MULTI_GPU'
downcast_bf16: 'no'
fsdp_config: {}
machine_rank: 0
main_process_ip: null
main_process_port: null
main_training_function: main
-mixed_precision: 'bf16'
+mixed_precision: 'fp8'
num_machines: 1
num_processes: 1
num_processes: 8
use_cpu: false
```
