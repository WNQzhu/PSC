CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-64k.py \
       --scale 7b \
       --max_length 64k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/coursera.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/mlora_test.7b.lr-2e-4.bs4.3000step.pi.64k.base/checkpoint-1500 


CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-64k.py \
       --scale 7b \
       --max_length 64k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/codeU.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/mlora_test.7b.lr-2e-4.bs4.3000step.pi.64k.base/checkpoint-1500 

CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-64k.py \
       --scale 7b \
       --max_length 64k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/gsm100.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/mlora_test.7b.lr-2e-4.bs4.3000step.pi.64k.base/checkpoint-1500 


CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-64k.py \
       --scale 7b \
       --max_length 64k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/quality.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/mlora_test.7b.lr-2e-4.bs4.3000step.pi.64k.base/checkpoint-1500 

CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-64k.py \
       --scale 7b \
       --max_length 64k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/sci_fi.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/mlora_test.7b.lr-2e-4.bs4.3000step.pi.64k.base/checkpoint-1500 


CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-64k.py \
       --scale 7b \
       --max_length 64k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/topic_retrieval_longchat.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/mlora_test.7b.lr-2e-4.bs4.3000step.pi.64k.base/checkpoint-1500 


CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-64k.py \
       --scale 7b \
       --max_length 64k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/tpo.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/mlora_test.7b.lr-2e-4.bs4.3000step.pi.64k.base/checkpoint-1500 
