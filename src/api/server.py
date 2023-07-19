"""
FastAPI backend
"""

import nltk
import numpy as np
import torch
from fastapi import FastAPI
from nltk.tokenize import sent_tokenize
from pydantic import BaseModel
from transformers import pipeline

nltk.download("punkt")

MODEL_NAME_OR_PATH = "/var/model"  # mount in docker-compose

app = FastAPI()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
nlp = pipeline(
    "token-classification",
    model=MODEL_NAME_OR_PATH,
    tokenizer=MODEL_NAME_OR_PATH,
    ignore_labels=["O"],
    device=device,
)


def serialize_dict_with_np_float(preds: dict) -> dict:
    """
    Serialize dictionary with np.float32 values as float.
    Args:
    - preds (dict): A dictionary with np.float32 values.
    Returns:
    - preds (dict): A dictionary with float values for np.float32 keys.
    """
    for idx, item in enumerate(preds):
        for key, value in item.items():
            if isinstance(value, np.float32):
                preds[idx][key] = float(value)
    return preds


class TextRequest(BaseModel):
    text: str


@app.get("/")
def check_health():
    return {"Health check": "Ok"}


@app.post("/predict_ner")
def predict_ner(text_request: TextRequest):
    preds = nlp(sent_tokenize(text_request.text))
    # handle non-serializable np.float32 -> float
    preds = [serialize_dict_with_np_float(item) for item in preds]
    return {"response": preds}


# Run from the command line
# uvicorn src.api.main:app --reload --host=127.0.0.1 --port=8000
