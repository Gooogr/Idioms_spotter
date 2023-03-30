# Idioms spotter
🤗 Transformers for identification English potentials idiomatic expressions (PIE) in text.

## Dataset
Result dataset is available for download from the HuggingFace hub: [Dataset page](https://huggingface.co/datasets/Gooogr/pie_idioms)

Dataset based on MAGPIE and PIE corpuses:
* [magpie-corpus](https://github.com/hslh/magpie-corpus) 
* [pie-annotation](https://github.com/hslh/pie-annotation) 

Full data preparation pipeline available in [data_preparation](https://github.com/Gooogr/Idioms_spotter/blob/main/notebooks/data_preparation.ipynb) notebook.
To obtain the json source files, run these commands from the root of the project:
```
curl -o ./data/raw/pie-corpus.json https://raw.githubusercontent.com/hslh/pie-annotation/master/PIE_annotations_all_no_sentences.json
curl -o ./data/raw/magpie-corpus.jsonl https://raw.githubusercontent.com/hslh/magpie-corpus/master/MAGPIE_unfiltered.jsonl
```
Note: the PIE corps needs to be further enriched with data in order to obtain suggestions and context. [Details](https://github.com/hslh/pie-annotation#contents--usage)

Raw and already enriched data can be downloaded [here](https://drive.google.com/file/d/1Hvlqp3VU9DeiZeocJNzG4GaxGduOyFAG/view?usp=sharing).

## Supported models
Supported models for training: <br>
* BERT
* RoBERTa
* DistilBERT
* ConvBERT

In general, the list of models is determined by their support in the [AutoModelForTokenClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForTokenClassification) and AutoTokenizer classes.

## Fine-tuned models
* [xlm-roberta-base-pie](https://huggingface.co/Gooogr/xlm-roberta-base-pie)

## Training from scratch

Environment setup
```
conda create -n idioms python=3.9
conda activate idioms
pip install -r requirements.txt
# If you want to use additional project's notebooks
pip install notebook ipywidgets
```

The following example fine-tunes XLM-RoBERTa:
```
python3 ./src/train.py \
  --model_name_or_path xlm-roberta-base \
  --output_dir ./models/xlm-roberta-base-finetuned \
  --num_train_epochs 10 \
  --seed 42 \
  --per_device_train_batch_size 16 \
  --per_device_eval_batch_size 16 \
  --log_level info \
  --weight_decay 0.01 \
  --logging_steps 1000 \
  --load_best_model_at_end True \
  --metric_for_best_model f1 \
  --greater_is_better True \
  --do_train True \
  --evaluation_strategy epoch \
  --save_strategy epoch 
```
Code will either run on pre-saved model from selected folder or download model from huggingface.co/models based on model identifier.
Training can be continued from the selected checkpoint s well. Full list of training parameters available [here](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L135)

Alternatively, you can specify model and training params in `train.sh`
```
bash train.sh
```

## License
Distributed under the MIT License.
