"""
Fine-tuning the HF models for PIEs token classification.
"""
import os
import sys
import logging
import torch
import wandb
import numpy as np
from helper import tokenize_and_allign_labels, create_compute_metrics
from dataclasses import dataclass, field
import datasets
from datasets import load_dataset
import transformers
from transformers.trainer_utils import get_last_checkpoint
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

logger = logging.getLogger(__name__)
wandb.login()

if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataTrainArguments, ModelArguments))
    train_args, data_args, model_args = parser.parse_args_into_dataclasses()

    # Set seed 
    set_seed(train_args.seed)

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)]
    )

    if train_args.should_log:
        # The default of training_args.log_level is passive, so we set 
        # log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = train_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    logger.info(f"Training/evaluation parameters {train_args}")
    
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
    
    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(train_args.output_dir) and train_args.do_train and not train_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(train_args.output_dir)
        if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({train_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and train_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set up model
    model_config = AutoConfig.from_pretrained(
        model_args.model_name_or_path, 
        num_labels=tags.num_classes, 
        id2label=index2tag, 
        label2id=tag2index)

    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path, 
        config=model_config).to(device)

    # Set up trainer
    compute_metrics = create_compute_metrics(index2tag)
    trainer = Trainer(
        model = model,
        args=train_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=dataset_encoded['train'],
        eval_dataset=dataset_encoded['validation'],
        tokenizer=tokenizer)

    # Train model
    if train_args.do_train:
        logger.info("*** Train model ***")
        checkpoint = None
        if train_args.resume_from_checkpoint is not None:
            checkpoint = train_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model(output_dir=train_args.output_dir)

        if train_args.push_to_hub:
            trainer.push_to_hub()

    # Evaluate model
    if train_args.do_eval:
        logger.info("*** Evaluate model ***")
        eval_metrics = trainer.evaluate()
        logger.info(eval_metrics)





