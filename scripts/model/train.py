# pylint: disable=C0103,W0632,W0621,E0611

"""
Fine-tuning HF models for PIEs token classification.
Full list of TrainingArguments available here:
https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py
"""
import os

import huggingface_hub
import numpy as np
import pandas as pd
from arguments import DataTrainArguments, ModelArguments, PeftArguments
from datasets import load_dataset
from model import WeightedTrainer
from model_helper import (
    create_compute_metrics,
    get_device,
    get_logger,
    get_model_config,
    get_tags_classification_weights,
    tokenize_and_allign_labels,
)
from peft import LoraConfig, TaskType, get_peft_model
from report_helper import print_classification_report
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    DataCollatorForTokenClassification,
    HfArgumentParser,
    TrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint

import wandb

wandb.login()

if __name__ == "__main__":
    parser = HfArgumentParser(
        (TrainingArguments, DataTrainArguments, ModelArguments, PeftArguments)
    )
    train_args, data_args, model_args, peft_args = parser.parse_args_into_dataclasses()

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
    if peft_args.use_lora:
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=peft_args.r,
            lora_alpha=peft_args.lora_alpha,
            lora_dropout=peft_args.lora_dropout,
            bias=peft_args.bias,
        )
        model = get_peft_model(model, peft_config)

    model = model.to(device)

    # Set up tokenizer and collator
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
    trainer = WeightedTrainer(
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
            model = AutoModelForTokenClassification.from_pretrained(
                best_ckpt_path, config=model_config
            )
            repo_name = model_args.model_name_or_path.split("/")[-1]
            repo_name = f"{repo_name}-pie"
            model.push_to_hub(repo_name)
            tokenizer.push_to_hub(repo_name)

    # Evaluate model
    if train_args.do_eval:
        logger.info("*** Evaluate model ***")
        eval_metrics = trainer.evaluate()
        logger.info(eval_metrics)

        # Create confusion matrix for each class
        eval_data = dataset_encoded["validation"]
        true_ids = eval_data["labels"]
        predictions_padded = trainer.predict(eval_data).predictions
        predicted_ids_padded = np.argmax(predictions_padded, axis=2)

        predicted_ids = []
        for true_item, padded_item in zip(true_ids, predicted_ids_padded):
            predicted_ids.append(padded_item[: len(true_item)])

        report_df = pd.DataFrame({"true_ids": true_ids, "predicted_ids": predicted_ids})

        # Convert labels ids to labels itself. -100 will be marked as "IGN" (ignore) and removed
        report_df["true_labels"] = report_df["true_ids"].apply(
            lambda x: [index2tag.get(i, "IGN") for i in x]
        )
        report_df["predicted_labels"] = report_df["predicted_ids"].apply(
            lambda x: [index2tag.get(i, "IGN") for i in x]
        )
        tokens_df = report_df.explode(column=list(report_df.columns), ignore_index=True)
        tokens_df = tokens_df.query("true_labels != 'IGN'")

        print_classification_report(
            y_true=tokens_df["true_labels"],
            y_pred=tokens_df["predicted_labels"],
            labels=[
                label for _, label in sorted(index2tag.items(), key=lambda x: x[0])
            ],
            normalize="true",
        )
