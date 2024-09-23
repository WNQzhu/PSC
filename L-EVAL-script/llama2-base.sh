python3 Baselines/llama2-base.py \
       --scale 7b \
       --max_length 4k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/coursera.jsonl

python3 Baselines/llama2-base.py \
       --scale 7b \
       --max_length 4k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/codeU.jsonl

python3 Baselines/llama2-base.py \
       --scale 7b \
       --max_length 4k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/gsm100.jsonl

python3 Baselines/llama2-base.py \
       --scale 7b \
       --max_length 4k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/quality.jsonl


python3 Baselines/llama2-base.py \
       --scale 7b \
       --max_length 4k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/sci_fi.jsonl

python3 Baselines/llama2-base.py \
       --scale 7b \
       --max_length 4k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/topic_retrieval_longchat.jsonl


python3 Baselines/llama2-base.py \
       --scale 7b \
       --max_length 4k \
       --metric exam_eval \
       --model_path /mnt/wnq/model/meta-llama/llama2-7b-hf \
       --task_path LEval-data/Closed-ended-tasks/tpo.jsonl

echo "llama2-4k base done"

