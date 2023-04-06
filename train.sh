#!/bin/bash

: '
Model training script example. The code will either run on a pre-saved model from the 
selected folder or download a model from huggingface.co/models based on the 
model identifier. Training can be continued from the selected checkpoint as well. 
The full list of training parameters is available here:
https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L135
'

python3 ./src/model/train.py \
  --model_name_or_path xlm-roberta-base\
  --output_dir ./models/xlm-roberta-base-pie \
  --num_train_epochs 10 \
  --seed 42 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --log_level info \
  --weight_decay 0.01 \
  --logging_steps 250 \
  --load_best_model_at_end True \
  --metric_for_best_model eval_f1\
  --greater_is_better True \
  --do_train True \
  --do_eval True \
  --push_to_hub False \
  --report_to="wandb" \
  --evaluation_strategy epoch \
  --save_strategy epoch 