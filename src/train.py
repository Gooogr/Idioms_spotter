"""
Fine-tuning the HF models for PIEs token classification.
"""
import logging
import torch
import numpy as np
from helper import tokenize_and_allign_labels, compute_metrics
from dataclasses import dataclass, field
from datasets import load_dataset
from transformers import (
    HfArgumentParser,
    TrainingArguments,
    AutoTokenizer,
    AutoConfig,
    Trainer,
    DataCollatorForTokenClassification,
    AutoModelForTokenClassification,
    set_seed
)

# Full list of TrainingArguments available here
# https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py

@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )

@dataclass
class DataTrainArguments:
    dataset_name: str = field(
        default = 'Gooogr/pie_idioms',
        metadata={"help": "Dataset identifier from huggingface.co/datasets"}
    )

if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataTrainArguments, ModelArguments))
    train_args, data_args, model_args = parser.parse_args_into_dataclasses()

    # Set seed 
    set_seed(train_args.seed)

    # TODO: add logging
    
    # Load dataset and get tags for model config
    dataset = load_dataset(data_args.dataset_name)
    tags = dataset['train'].features['ner_tags'].feature
    index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

    # Device selection
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)
    data_collator = DataCollatorForTokenClassification(tokenizer) 

    # Convert word tokens -> sentencepice tokens
    dataset_encoded = dataset.map(
        tokenize_and_allign_labels, 
        batched=True, 
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=['ner_tags', 'tokens', 'idiom', 'is_pie'])

    # Set up model
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, 
        num_labels=tags.num_classes, 
        id2label=index2tag, 
        label2id=tag2index)

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path, 
        config=model_config).to(device)

    # Set up training parameters
    logging_steps = len(dataset_encoded['train']) // train_args.per_device_train_batch_size
    
    training_args = TrainingArguments(
        output_dir=train_args.output_dir,
        log_level=train_args.log_level,
        num_train_epochs=train_args.num_train_epochs,
        per_device_train_batch_size=train_args.per_device_train_batch_size,
        per_device_eval_batch_size=train_args.per_device_eval_batch_size,
        evaluation_strategy=train_args.evaluation_strategy,
        save_strategy=train_args.save_strategy,
        weight_decay=train_args.weight_decay,
        logging_steps=logging_steps,
        load_best_model_at_end=train_args.load_best_model_at_end,
        metric_for_best_model=train_args.metric_for_best_model,
        greater_is_better=train_args.greater_is_better
    )

    # Train model
    if train_args.do_train:
        trainer = Trainer(
            model = model,
            args=training_args,
            data_collator=data_collator,
            compute_metrics=compute_metrics,
            train_dataset=dataset_encoded['train'],
            eval_dataset=dataset_encoded['validation'],
            tokenizer=tokenizer
        )
        trainer.train()


