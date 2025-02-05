```
python examples/T5/run_t5_flax.py \
	--output_dir="./norwegian-t5-base" \
	--model_type="t5" \
	--config_name="google-t5/t5-small" \
	--tokenizer_name="google-t5/t5-small" \
	--dataset_name="oscar" \
	--max_seq_length="512" \
	--per_device_train_batch_size="2" \
	--per_device_eval_batch_size="2" \
	--adafactor \
	--learning_rate="0.005" \
	--weight_decay="0.001" \
	--warmup_steps="2000" \
	--overwrite_output_dir \
	--logging_steps="500" \
	--save_steps="10000" \
	--eval_steps="2500" \
	--push_to_hub \
    --use_data_sample 
```