#!/bin/bash

: '
Model evaluation script example. The code will either run on a pre-saved model from the 
selected folder or download a model from huggingface.co/models based on the 
model identifier.
'
# Note: --output_dir not used in evaluation, but required for Trainer.

python3 scripts/model/train.py \
  --model_name_or_path models/xlm-roberta-base-pie \
  --output_dir ./models/xlm-roberta-base-pie \
  --seed 42 \
  --per_device_eval_batch_size 16 \
  --log_level info \
  --do_train False \
  --do_eval True \
  --push_to_hub False \
  --report_to="wandb" \
  --evaluation_strategy epoch \
  --save_strategy epoch \
  --no_cuda True \
  --use_lora False \