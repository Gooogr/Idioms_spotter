# Idioms spotter
🤗 Transformers for identification English potentials idiomatic expressions (PIE) in text.

![Alt text](/references/api_example.png?raw=true)

## Dataset
Result dataset is available for download from the Hugging Face hub: <br>
[Dataset page](https://huggingface.co/datasets/Gooogr/pie_idioms)

The dataset is based on MAGPIE and PIE corpuses:
* [magpie-corpus](https://github.com/hslh/magpie-corpus) 
* [pie-annotation](https://github.com/hslh/pie-annotation) 

The full data preparation pipeline available in [data_preparation](https://github.com/Gooogr/Idioms_spotter/blob/main/notebooks/data_preparation.ipynb) notebook. To obtain the json source files, run:
```
bash scripts/download_raw_datasets.sh 
```
Note that the PIE corpus needs to be further enriched with data in order to obtain suggestions and context. More details can be found [here](https://github.com/hslh/pie-annotation#contents--usage)

You can download both raw and already enriched data from [here](https://drive.google.com/file/d/1Hvlqp3VU9DeiZeocJNzG4GaxGduOyFAG/view?usp=sharing).

## Supported models
The list of supported for training models is determined by their support in the [AutoModelForTokenClassification](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForTokenClassification) and AutoTokenizer classes.

For example, the following models are trainable: <br>
* BERT
* RoBERTa
* DistilBERT
* ConvBERT

## Fine-tuned models
The following fine-tuned models are available on Hugging Face model hub:
| Model                                                                  | Loss  | F1    | Precision | Recall |
|------------------------------------------------------------------------|-------|-------|-----------|--------|
| [XLM Roberta base](https://huggingface.co/Gooogr/xlm-roberta-base-pie) | 0.095 | 0.856 | 0.836     | 0.876  |
|                                                                        |       |       |           |        |

All metrics are obtained on the validation part of the dataset

## Training from scratch

To set up the environment, follow these steps:
```
conda create -n idioms python=3.8
conda activate idioms
pip3 install poetry==1.5.1
poetry install --only main
```

The following example shows how to fine-tune XLM-RoBERTa:
```
python3 ./sripts/model/train.py \
  --model_name_or_path xlm-roberta-base\
  --output_dir ./models/xlm-roberta-base-pie \
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
  --do_eval True \
  --report_to="wandb" \
  --evaluation_strategy epoch \
  --save_strategy epoch 
```
The code will either run on a pre-saved model from the selected folder or download a model from huggingface.co/models based on the model identifier. Training can be continued from the selected checkpoint as well. The full list of training parameters is available [here](https://github.com/huggingface/transformers/blob/main/src/transformers/training_args.py#L135)

Alternatively, you can specify model and training params in `train.sh`
```
bash train.sh
```

## Running web app with API
The application consists of two containers:
* Model backend based on FastAPI
* Web application build on Streamlit

You can specify model folder manually inside `docker-compose.yaml`and run
```
docker-compose up --build
```

Alternatively, you can use `run_api.sh` <br>
```
bash run_api.sh <model_name_or_path> [<force_download>]
```
Where:
* model_name_or_path - path to model folder or model id in the HuggingFace hub
* force_download - optional parameter (default is False). If True, the model will be forcibly downloaded from the hub, even if it has already been saved.

For example:
```
chmod +x run_app.sh
bash run_app.sh Gooogr/xlm-roberta-base-pie 
```

## License
Distributed under the MIT License.
