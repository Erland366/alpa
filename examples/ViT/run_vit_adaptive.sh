python run_image_classification_adaptive.py \
    --output_dir ./vit-base-patch16-imagenette \
    --model_name_or_path google/vit-base-patch16-224-in21k \
    --train_dir /aios-store/akhmed-rampart/imagenet/train \
    --validation_dir imagenette2/val \
    --num_train_epochs 10000000 \
    --num_micro_batches 2 \
    --learning_rate 1e-5 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 64 \
    --overwrite_output_dir \
    --preprocessing_num_workers 32 \
    --pretrain=False \
    --scale_lr=False