# PSC: Extending Context Window of Large Language Models via Phase Shift Calibration

The PSC module is implemented in the 'lora_llama.py' file, with the class name 'Phase'.
We can turn on/off the PSC/LoRA by setting 'using_lora' and 'using_phase' to True/False.


## Parameter initialization
We first need to generate the initialization parameters for the PSC/LoRA module.
```
output_dir=./PSC_output
mkdir -p ${output_dir}
python3 phase.py --checkpoint_path ${output_dir}
```

## Fine-tuning
```
name="phase_shift_calibration"
dataset_name="sampled_rpj.jsonl"
lr=2e-4
seq_len=65536
yarn_factor=16.0
target=${output_dir}
CUDA_VISIBLE_DEVICES=3,2,1,0 torchrun --nproc_per_node 4 \
                             --master_port 15577 \
			     cotrain_main.py \
			     --model_name_or_path ${base_llama_model_7b_hf} \
			     --train_file ${dataset_name} \
			     --lora_dir ${target}
			     --bf16 True \
			     --trainable_params "lora,phase" \
			     --output_dir ./${output_dir} \
			     --yarn_factor ${yarn_factor} \
			     --cache_dir ./cache-debug-tmp \
			     --model_max_length ${seq_len} \
			     --use_flash_attn True \
			     --low_rank_training True \
			     --per_device_train_batch_size 1 \
			     --per_device_eval_batch_size 1 \
			     --gradient_accumulation_steps 4 \
			     --evaluation_strategy "no" \
			     --save_strategy "steps" \
			     --save_steps 500 \
			     --save_total_limit 6 \
			     --learning_rate ${lr} \
			     --weight_decay 0.0 \
			     --warmup_steps 20 \
			     --lr_scheduler_type "constant_with_warmup" \
			     --seed 42 \
			     --tf32 True \
			     --max_steps 3000 \
			     --ddp_find_unused_parameters False 
		
```