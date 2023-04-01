'''
Helper functions for Token Classification tasks
'''
import numpy as np
from transformers import PreTrainedTokenizer
from datasets.formatting.formatting import LazyBatch
from seqeval.metrics import f1_score, accuracy_score, recall_score, precision_score

def tokenize_and_allign_labels(
    examples: LazyBatch, 
    tokenizer: PreTrainedTokenizer
) -> LazyBatch:
    '''
    Apply sentencepiece tokenization to input batch, then allign new tokens with 
    NER tags. Input assumes that source string was pre-tokenized by words.
    Args:
        examples (LazyBatch): Input batch
        tokenizer (HuggingFace): tokenizer
    Returns:
        tokenized_inputs (LazyBatch): Processed batch, ready for NER model training 
    '''
    tokenized_inputs = tokenizer(examples['tokens'], 
                                 truncation=True, 
                                 is_split_into_words=True)
    labels = []
    for idx, label in enumerate(examples['ner_tags']):
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
    tokenized_inputs['labels'] = labels
    return tokenized_inputs

#----------------------------------------

def allign_predictions(predictions, label_ids, index2tag):
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

def create_compute_metrics(index2tag):
    def compute_metrics(eval_pred):
        nonlocal index2tag
        y_pred, y_true = allign_predictions(
            eval_pred.predictions, 
            eval_pred.label_ids,
            index2tag)
        return {'f1': f1_score(y_true, y_pred), 
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred),
                'recall': recall_score(y_true, y_pred)}
    return compute_metrics
