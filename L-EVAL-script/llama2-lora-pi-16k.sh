CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-16k.py \
       --scale 7b \
       --max_length 16k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/coursera.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/pi_16k_base_ck_500


CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-16k.py \
       --scale 7b \
       --max_length 16k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/codeU.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/pi_16k_base_ck_500

CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-16k.py \
       --scale 7b \
       --max_length 16k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/gsm100.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/pi_16k_base_ck_500


CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-16k.py \
       --scale 7b \
       --max_length 16k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/quality.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/pi_16k_base_ck_500

CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-16k.py \
       --scale 7b \
       --max_length 16k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/sci_fi.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/pi_16k_base_ck_500


CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-16k.py \
       --scale 7b \
       --max_length 16k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/topic_retrieval_longchat.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/pi_16k_base_ck_500


CUDA_VISIBLE_DEVICES=1 python3  Baselines/llama2-ft-pi-16k.py \
       --scale 7b \
       --max_length 16k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/tpo.jsonl \
       --peft_model /mnt/wnq/new/rec/phase/pi_16k_base_ck_500
