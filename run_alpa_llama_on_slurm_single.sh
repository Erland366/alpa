#!/bin/bash
#SBATCH --job-name=alpa_llama_test_single
#SBATCH -p llm_s
#SBATCH -N 1
#SBATCH --gres=gpu:8

# Start Ray on HEAD
# set object_spilling_threshold to avoid ray object spilling out of disk errors
ray start --head  --system-config='{"object_spilling_threshold":0.99}'

# Run Alpa test
python3 -m alpa.test_install

cd $HOME/alpa/examples/llama_finetune

# git clone https://github.com/zigzagcai/EasyLM
export PYTHONPATH=$HOME/EasyLM:$PYTHONPATH

# ShareGPT dataset
# https://github.com/lm-sys/FastChat/issues/90#issuecomment-1493250773

python3 run_easylm_flax.py \
    --output_dir="./output" \
    --model_name_or_path="huggyllama/llama-7b" \
    --dataset_name="$HOME/alpa_data/data/sg_90k_part1.json" \
    --do_train \
    --block_size="1024" \
    --per_device_train_batch_size="32" \
    --per_device_eval_batch_size="16" \
    --num_micro_batches 32 \
    --operator_parallel 2 \
    --pipeline_parallel 2 \
    --dtype="float16" \
    --learning_rate="5e-4" --warmup_ratio="0.03" \
    --weight_decay="0.0" \
    --overwrite_output_dir \
    --num_train_epochs="3" \
    --logging_steps="1" \
    --save_steps="3000" \
    --eval_steps="1000"

cd $HOME/alpa

# Optional. Slurm will terminate all processes automatically
ray stop
