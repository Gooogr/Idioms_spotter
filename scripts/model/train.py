# pylint: disable=C0103,W0632,W0621,E1102

"""
Fine-tuning HF models for PIEs token classification.
Full list of TrainingArguments available here:
https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
"""
import os
from dataclasses import dataclass, field

import huggingface_hub
import torch
from datasets import load_dataset
from model_helper import (
    create_compute_metrics,
    get_device,
    get_logger,
    get_model_config,
    get_tags_classification_weights,
    tokenize_and_allign_labels,
)
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    Trainer,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import wandb

wandb.login()


@dataclass
class ModelArguments:
    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"
        },
    )


@dataclass
class DataTrainArguments:
    dataset_name: str = field(
        default="Gooogr/pie_idioms",
        metadata={"help": "Dataset identifier from huggingface.co/datasets"},
    )


class CustomTrainer(Trainer):
    def __init__(self, class_weights, device, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights
        self.device = device

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        labels = inputs.get("labels")
        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")
        # compute custom loss
        loss_fct = torch.nn.CrossEntropyLoss(
            weight=torch.tensor(self.class_weights)
        ).to(self.device)
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


if __name__ == "__main__":
    parser = HfArgumentParser((TrainingArguments, DataTrainArguments, ModelArguments))
    train_args, data_args, model_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logger = get_logger(train_args)
    logger.info(f"Training/evaluation parameters {train_args}")

    # Set seed
    set_seed(train_args.seed)

    # Device selection
    device = get_device(train_args)

    # Load dataset and encode tags
    dataset = load_dataset(data_args.dataset_name)
    tags = dataset["train"].features["ner_tags"].feature
    index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

    # Set up model
    model_config = get_model_config(model_args.model_name_or_path, tags)
    model = AutoModelForTokenClassification.from_pretrained(
        model_args.model_name_or_path, config=model_config
    )
    model = model.to(device)

    # Set up tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path)

    data_collator = DataCollatorForTokenClassification(tokenizer)

    # Convert word tokens -> sentencepice tokens
    dataset_encoded = dataset.map(
        tokenize_and_allign_labels,
        batched=True,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=["ner_tags", "tokens", "idiom", "is_pie"],
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(train_args.output_dir)
        and train_args.do_train
        and not train_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(train_args.output_dir)
        if last_checkpoint is None and len(os.listdir(train_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({train_args.output_dir}) already exists "
                "and is not empty. Use --overwrite_output_dir to overcome."
            )
        if last_checkpoint is not None and train_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}."
                " To avoid this behavior, change the `--output_dir` or add "
                "`--overwrite_output_dir` to train from scratch."
            )

    # Set up trainer
    compute_metrics = create_compute_metrics(index2tag)
    weights = get_tags_classification_weights(dataset["train"], "ner_tags")
    trainer = CustomTrainer(
        model=model,
        args=train_args,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        train_dataset=dataset_encoded["train"],
        eval_dataset=dataset_encoded["validation"],
        tokenizer=tokenizer,
        class_weights=weights,
        device=device,
    )

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
            huggingface_hub.login()
            # To make sure that we push best, not last model
            best_ckpt_path = trainer.state.best_model_checkpoint
            classification_model = AutoModelForTokenClassification.from_pretrained(
                best_ckpt_path, config=model_config
            )
            repo_name = model_args.model_name_or_path.split("/")[-1]
            repo_name = f"{repo_name}-pie"
            classification_model.push_to_hub(repo_name)
            tokenizer.push_to_hub(repo_name)

    # Evaluate model
    if train_args.do_eval:
        logger.info("*** Evaluate model ***")
        eval_metrics = trainer.evaluate()
        logger.info(eval_metrics)
