"""
Fine-tuning the HF models for PIEs token classification.
"""
from dataclasses import dataclass, field
from transformers import (
    HfArgumentParser,
    TrainingArguments
)

# Full list of TrainingArguments available here
# https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, ModelArguments))
    train_args, model_args = parser.parse_args_into_dataclasses()
    # print(train_args.__dict__)

## List of params
# model_name_or_path
# num_train_epochs
# seed 
# per_device_train_batch_size
# per_device_eval_batch_size

