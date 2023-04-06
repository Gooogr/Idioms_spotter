'''
An auxiliary script for downloading and saving a pre-trained model for its
further inference in a docker container.
'''
import os
import argparse
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification

logging.basicConfig(level=logging.INFO)


def save_model(model_id: str, save_dir: str):
    # Load the model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForTokenClassification.from_pretrained(model_id)

    # Create the save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save the model and tokenizer to the specified directory
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Save a HF model and tokenizer to a specified folder')
    parser.add_argument(
        '-m',
        '--model_id',
        type=str,
        help='The identifier of the pre-trained model to load')
    parser.add_argument(
        '-d',
        '--save_dir',
        type=str,
        required=False,
        default=None,
        help='The directory to save the model and tokenizer in')

    args = parser.parse_args()

    # by default - save to /model/model_id folder
    if args.save_dir is None:
        save_dir = os.path.join('./models/', args.model_id.split('/')[-1])
    else:
        save_dir = args.save_dir

    save_model(args.model_id, save_dir)
    logging.info(f'Succesfully saved model and tokenizer in {save_dir}')
