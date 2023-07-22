"""
Helper functions for Token Classification tasks
"""
import logging
import sys
from typing import Callable, Dict, List, Tuple, Union

import numpy as np
import torch
import transformers
from datasets import ClassLabel, Dataset
from datasets.formatting.formatting import LazyBatch
from seqeval.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import AutoConfig, PreTrainedTokenizer, TrainingArguments
from transformers.trainer_utils import EvalPrediction


def tokenize_and_allign_labels(
    examples: LazyBatch, tokenizer: PreTrainedTokenizer
) -> LazyBatch:
    """
    Apply sentencepiece tokenization to input batch, then allign new tokens
    with NER tags. Input assumes that source string was pre-tokenized by words.
    Args:
    - examples (LazyBatch): Input batch
    - tokenizer (HuggingFace): tokenizer
    Returns:
    - tokenized_inputs (LazyBatch): Processed batch, ready for model training
    """
    tokenized_inputs = tokenizer(
        examples["tokens"], truncation=True, is_split_into_words=True
    )
    labels = []
    for idx, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=idx)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None or word_idx == previous_word_idx:
                label_ids.append(-100)
            else:
                label_ids.append(label[word_idx])
            previous_word_idx = word_idx
        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def allign_predictions(
    predictions: np.ndarray, label_ids: np.ndarray, index2tag: dict
) -> Tuple[List[List[str]], List[List[str]]]:
    """
    Given a batch of model predictions and corresponding true label IDs, align
    the predicted and true labels based on the non-padding and non-masked
    tokens in the input sequence.

    Args:
    - predictions (np.ndarray): A batch of model predictions.
        Shape (batch_size, seq_len, num_labels).
    - label_ids (np.ndarray): A batch of true label IDs.
        Shape (batch_size, seq_len).
    - index2tag (dict): A dictionary that maps label indices to label names.

    Returns:
    - pred_list (List[List[str]]): A list of predicted labels for each example
        in the batch. Shape (batch_size, variable).
    - label_list (List[List[str]]): A list of true labels for each example
        in the batch. Shape (batch_size, variable).
    """
    preds = np.argmax(predictions, axis=2)
    batch_size, seq_len = preds.shape
    label_list, pred_list = [], []

    for batch_idx in range(batch_size):
        example_labels, example_preds = [], []
        for seq_idx in range(seq_len):
            if label_ids[batch_idx, seq_idx] != -100:
                example_labels.append(index2tag[label_ids[batch_idx, seq_idx]])
                example_preds.append(index2tag[preds[batch_idx, seq_idx]])
        label_list.append(example_labels)
        pred_list.append(example_preds)
    return pred_list, label_list


def create_compute_metrics(
    index2tag: dict,
) -> Callable[[EvalPrediction], Dict[str, float]]:
    """
    Create and return a function for computing evaluation metrics for a
    sequence tagging model, given a mapping from label indices to label names.

    Args:
    - index2tag (dict): A dictionary that maps label indices to label names.

    Returns:
    - compute_metrics (Callable): A function that takes an `EvalPrediction`
        object as input and returns a dictionary of evaluation metrics,
        including F1 score, accuracy, precision, and recall.
    """

    def compute_metrics(eval_pred: EvalPrediction) -> Dict[str, float]:
        nonlocal index2tag
        y_pred, y_true = allign_predictions(
            eval_pred.predictions, eval_pred.label_ids, index2tag
        )
        return {
            "f1": f1_score(y_true, y_pred),
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred),
            "recall": recall_score(y_true, y_pred),
        }

    return compute_metrics


def get_tags_classification_weights(
    dataset: Dataset, labels_key: str = "ner_tags"
) -> List[float]:
    """
    Calculate class weights based on the number of objects in each class.

    Args:
    - dataset (Dataset): input Hugging Face Dataset object.
    - labels_key (str): key for NER tags inside Dataset.

    Returns:
    - List[float]: list with weigts for each unique NER tag.
    """
    all_tags = dataset[labels_key]
    # Count amount of each NER tag
    count_dict: Dict[int, float] = {}
    for item in all_tags:
        for tag in item:
            count_dict[tag] = count_dict.get(tag, 0) + 1
    # Count total tags amount
    total_amount = sum(count_dict.values())
    # Calculate weights
    result = [total_amount / item for item in count_dict.values()]
    return result


def get_logger(train_args: TrainingArguments) -> logging.Logger:
    """
    Set up root logger based on TrainingArguments.

    Args:
    - train_args(TrainingArguments): The TrainingArguments object containing training settings.

    Returns:
    - Logger object with selected information level and format
    """
    log_level = train_args.get_process_log_level()
    transformers.utils.logging.set_verbosity(log_level)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=log_level,
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    return logging.getLogger(__name__)


def get_device(train_args: TrainingArguments) -> Union[str, torch.device]:
    """
    Determines and returns the appropriate device for training based on the TrainingArguments.

    Args:
    - train_args (TrainingArguments): The TrainingArguments object containing training settings.

    Returns:
    - Union[str, torch.device]: The selected device.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if train_args.no_cuda:
        device = torch.device("cpu")
    return device


def get_model_config(model_name_or_path: str, tags: ClassLabel) -> AutoConfig:
    """
    Loads the model configuration for a given model name or path and a set of tags (class labels).

    Args:
    - model_name_or_path (str): The HuggingFace hub ID or path of the pre-trained model
        to load the configuration for.
    - tags (ClassLabel): A ClassLabel object containing the list of class labels for the model.

    Returns:
    - AutoConfig: The AutoConfig object containing the loaded model configuration.

    Example:
        >>> tags = ClassLabel(names=["positive", "negative", "neutral"], num_classes=3)
        >>> config = get_model_config("bert-base-uncased", tags)
    """
    index2tag = {idx: tag for idx, tag in enumerate(tags.names)}
    tag2index = {tag: idx for idx, tag in enumerate(tags.names)}

    model_config = AutoConfig.from_pretrained(
        model_name_or_path,
        num_labels=tags.num_classes,
        id2label=index2tag,
        label2id=tag2index,
    )
    return model_config
