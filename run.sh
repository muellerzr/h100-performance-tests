CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file fp8_single.yml bloomz-3b/accelerate_script.py
CUDA_VISIBLE_DEVICES="0" accelerate launch --config_file bf16_single.yml bloomz-3b/accelerate_script.py
accelerate launch --config_file bf16.yml bloomz-3b/accelerate_script.py
accelerate launch --config_file fp8.yml bloomz-3b/accelerate_script.py
