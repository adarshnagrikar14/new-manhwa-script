export KAGGLE_PREFIX="."

export LOG_PATH="$KAGGLE_PREFIX/models/logs"
export MODEL_DIR="black-forest-labs/FLUX.1-dev"
export CONFIG="$KAGGLE_PREFIX/default_config.yaml"
export OUTPUT_DIR="$KAGGLE_PREFIX/models/style_model"
export TRAIN_DATA="$KAGGLE_PREFIX/data/train/train.jsonl"

CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file $CONFIG train.py \
    --pretrained_model_name_or_path $MODEL_DIR \
    --cond_size=512 \
    --noise_size=512 \
    --subject_column="None" \
    --spatial_column="source" \
    --target_column="target" \
    --caption_column="caption" \
    --ranks 128 \
    --network_alphas 128 \
    --output_dir=$OUTPUT_DIR \
    --logging_dir=$LOG_PATH \
    --mixed_precision="bf16" \
    --train_data_dir=$TRAIN_DATA \
    --learning_rate=1e-4 \
    --train_batch_size=1 \
    --validation_prompt "K-pop manhwa style digital illustration of this image" \
    --num_train_epochs=300 \
    --validation_steps=300 \
    --checkpointing_steps=300 \
    --spatial_test_images "$KAGGLE_PREFIX/data/test/test_one.png" \
    --subject_test_images None \
    --test_h 512 \
    --test_w 512 \
    --num_validation_images=1