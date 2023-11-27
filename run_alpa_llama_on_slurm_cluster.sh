#!/bin/bash
#SBATCH --job-name=alpa_llama_test_cluster
#SBATCH -p llm_s
#SBATCH -N 2
#SBATCH -n 2
#SBATCH --gres=gpu:8

# Get names of nodes assigned
nodes=$(scontrol show hostnames "$SLURM_JOB_NODELIST")
nodes_array=($nodes)

# By default, set the first node to be head_node on which we run HEAD of Ray
head_node=${nodes_array[0]}
head_node_ip=$(srun -n 1 -N 1 -w "$head_node" hostname --ip-address)

# Setup port and variables needed
port=6379
ip_head=$head_node_ip:$port
export ip_head
# Start HEAD in background of head node
srun -n 1 -N 1 -w "$head_node" \
        ray start --head --node-ip-address="$head_node_ip" --port=$port --system-config='{"object_spilling_threshold":0.99}' --block &

# Optional, sometimes needed for old Ray versions
sleep 10

# Start worker nodes
# Number of nodes other than the head node
worker_num=$((SLURM_JOB_NUM_NODES - 1))
# Iterate on each node other than head node, start ray worker and connect to HEAD
for ((i = 1; i <= worker_num; i++)); do
    node_i=${nodes_array[$i]}
    echo "Starting WORKER $i at $node_i"
    srun -n 1 -N 1 -w "$node_i" \
        ray start --address "$ip_head" --block &
    sleep 5
done

# Run Alpa test
python3 -m alpa.test_install

cd $HOME/alpa/examples/llama_finetune

# EasyLM: 9757be87571e714da83f9311531c81db47953f63
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
    --pipeline_parallel 1 \
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
