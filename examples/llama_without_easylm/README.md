<!---
Copyright 2021 The HuggingFace Team. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->

# Fine-tuning OPT Language Models

## Instructions

### Launch a Ray cluster

1. Use the command below to launch ray on a head node  
  ```ray start --head```
2. (Optional) If you have more nodes, connect them to the head node. The command should look like this, but with the ip address and password printed by the previous command.   
  ```ray start --address='172.31.34.216:6379' --redis-password='5241590000000000'```

### Run training

**Note**: The command below is tested on AWS p3.16xlarge instances with 8 x 16GB V100 GPUs.
To run on other clusters, please tune the arguments `per_device_train_batch_size/num_micro_batches/operator_parallel/pipeline_parallel` to avoid out-of-memory and achieve a good throughput.
```
python3 run_clm_flax.py \
    --output_dir="./output" \
    --model_name_or_path="facebook/opt-2.7b" \
    --dataset_name="wikitext" \
    --dataset_config_name="wikitext-2-raw-v1" \
    --do_train --do_eval \
    --block_size="1024" \
    --per_device_train_batch_size="20" \
    --per_device_eval_batch_size="20" \
    --num_micro_batches 4 \
    --operator_parallel 4 \
    --pipeline_parallel 1 \
    --dtype="float16" \
    --learning_rate="5e-4" --warmup_steps="2000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="8" \
    --logging_steps="16" \
    --save_steps="2500" \
    --eval_steps="2500"
```

More documentation coming soon.

```
python3 examples/llama_without_easylm/run_llama_flax.py \
    --output_dir ./output \
    --model_name_or_path  Erland/Llama-3.2-1B-JAX \
    --dataset_name /fs-computility/llm/caizheng/alpa_data/data/sg_90k_part1.json \
    --do_train \
    --block_size 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --dtype float16 \
    --learning_rate 5e-4 \
    --weight_decay 0.0 \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 3000 \
    --eval_steps 1000 \
    --use_data_sample 
```

```
python examples/llama_without_easylm/run_llama_flax.py \
    --output_dir ./output \
    --config_name  Erland/Llama-3.2-1B-JAX \
    --dataset_name /fs-computility/llm/caizheng/alpa_data/data/sg_90k_part1.json \
    --do_train \
    --block_size 1024 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 2 \
    --dtype float16 \
    --learning_rate 5e-4 \
    --weight_decay 0.0 \
    --overwrite_output_dir \
    --num_train_epochs 3 \
    --logging_steps 1 \
    --save_steps 3000 \
    --eval_steps 1000 \
    --use_data_sample 
```


# Acknowledgement
Adopted from https://github.com/huggingface/transformers/tree/main/examples/flax/language-modeling
