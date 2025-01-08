python3 run_clm_flax.py \
    --output_dir="./output" \
    --model_name_or_path="Erland/Llama-3.2-1B-JAX" \
    --dataset_name="wikitext" \
    --dataset_config_name="wikitext-2-raw-v1" \
    --do_train --do_eval \
    --block_size="1024" \
    --per_device_train_batch_size="20" \
    --per_device_eval_batch_size="20" \
    --num_micro_batches 4 \
    --operator_parallel 1 \
    --pipeline_parallel 1 \
    --dtype="float16" \
    --learning_rate="5e-4" --warmup_steps="2000" \
    --adam_beta1="0.9" --adam_beta2="0.98" --weight_decay="0.01" \
    --overwrite_output_dir \
    --num_train_epochs="8" \
    --logging_steps="16" \
    --save_steps="2500" \
    --eval_steps="2500"
