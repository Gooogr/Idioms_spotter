python3 ./src/train.py \
  --model_name_or_path xlm-roberta-base \
  --output_dir ./models/xlm-roberta-base-finetuned \
  --num_train_epochs 10 \
  --seed 42 \
  --per_device_train_batch_size 16\
  --per_device_eval_batch_size 16\