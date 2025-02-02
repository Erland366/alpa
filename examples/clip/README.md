```bash
python examples/clip/run_clip.py \
    --output_dir ./clip-roberta-finetuned \
    --model_name_or_path ./clip-roberta \
    --data_dir $PWD/data \
    --dataset_name ydshieh/coco_dataset_script \
    --dataset_config_name=2017 \
    --image_column image_path \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train  --do_eval \
    --per_device_train_batch_size="64" \
    --per_device_eval_batch_size="64" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --push_to_hub
```

or if you want to try to train smaller dataset first 

```bash
python examples/clip/run_clip_torch.py \
    --output_dir ./clip-roberta-finetuned \
    --config_name openai/clip-vit-base-patch32 \
    --dataset_name "RIW/small-coco" \
    --image_column image \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train \
    --per_device_train_batch_size="2" \
    --per_device_eval_batch_size="2" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --max_train_samples 100
```

Or using the Flax version

```bash
python examples/clip/run_clip_flax.py \
    --output_dir ./clip-roberta-finetuned \
    --config_name openai/clip-vit-base-patch32 \
    --image_column image \
    --caption_column caption \
    --remove_unused_columns=False \
    --do_train \
    --per_device_train_batch_size="2" \
    --per_device_eval_batch_size="2" \
    --learning_rate="5e-5" --warmup_steps="0" --weight_decay 0.1 \
    --overwrite_output_dir \
    --max_train_samples 100 \
    --use-data-sample \
    --pipeline_parallel 2 \
    --operator_parallel 1
```