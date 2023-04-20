#!/bin/bash

model_name_or_path="$1"
force_download=${2:-false}

# Check if model name or path is not empty
if [ -z "$model_name_or_path" ]
then
    echo "model_name_or_path is empty"
    exit 1
fi

# Validate input pathes
if [[ ! "$force_download" =~ ^(true|false)$ ]]
then
    echo "force_download must be a boolean value or empty.\
Empty value will be set as False"
    exit 1
fi

echo "Model name or path: $model_name_or_path"
echo "Force download: $force_download"
echo ""

# Check if path exist
if [[ -d "$model_name_or_path" ]]
then
    echo "Warning! Foler $model_name_or_path exists on your filesystem. \
Thus, model downloading would be skipped. If you want to force re-download \
of the model from the Hugging Face model hub, use force_download=True"
fi

# Create model path based on model hub id link
# For example for Gooor/xlm-roberta-base-pie path would be ./models/xlm-roberta-base-pie
model_folder="$model_name_or_path"
if [[ ! -d "$model_name_or_path" ]]
then
    model_id=$(echo "$model_name_or_path" | awk -F '/' '{print $NF}')
    model_folder="./models/$model_id"
    echo "Model path for docker mount: $model_folder"
fi


# Download model
# Check possible folders existence or force_download flag
if [[ ( ! -d "$model_name_or_path" && ! -d "$model_folder" ) || "$force_download" = true ]]
then
    python3 ./src/scripts/download_model_from_hub.py -m $model_name_or_path
fi


# Pass folder path and run docker-compose 
MODEL_PATH=$model_folder docker-compose up --build

# How to run:
# bash run_api.sh <model_name_or_path> [<force_download>]