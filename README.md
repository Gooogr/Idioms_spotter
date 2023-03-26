# Idioms_spotter
🤗 Transformers for identification English potentials idiomatic expressions (PIE) in text.

## Model
Supported models for training: <br>
* BERT
* RoBERTa
* DistilBERT
* ConvBERT

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



