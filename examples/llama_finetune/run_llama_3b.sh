#!/bin/bash

# git clone https://github.com/zigzagcai/EasyLM
export PYTHONPATH=$HOME/EasyLM:$PYTHONPATH

# ShareGPT dataset
# https://github.com/lm-sys/FastChat/issues/90#issuecomment-1493250773

python3 examples/llama_finetune/run_easylm_flax.py \
    --output_dir="./output" \
    --model_name_or_path="meta-llama/Llama-3.2-3B" \
    --dataset_name="/fs-computility/llm/caizheng/alpa_data/data/sg_90k_part1.json" \
    --do_train \
    --block_size="1024" \
    --per_device_train_batch_size="2" \
    --per_device_eval_batch_size="2" \
    --num_micro_batches 32 \
    --operator_parallel 1 \
    --pipeline_parallel 1 \
    --dtype="bfloat16" \
    --learning_rate="5e-4" --warmup_ratio="0.03" \
    --weight_decay="0.0" \
    --overwrite_output_dir \
    --num_train_epochs="3" \
    --logging_steps="1" \
    --save_steps="3000" \
    --eval_steps="1000" \
    --architecture="llama3.2_3b" \
    --preprocessing_num_worker="8" \
    --max_train_samples="10" \
    --use_data_sample 
