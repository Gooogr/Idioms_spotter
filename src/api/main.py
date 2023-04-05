from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import numpy as np
import torch

MODEL_NAME = 'Gooogr/xlm-roberta-base-pie'


class TextRequest(BaseModel):
    text: str


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
nlp = pipeline("token-classification",
               model=MODEL_NAME,
               tokenizer=MODEL_NAME,
               device=device)

app = FastAPI()


@app.get("/")
def check_health():
    return {"Health check": "Ok"}


@app.post("/predict_ner")
def predict_ner(text_request: TextRequest):
    predictions = nlp(text_request.text)
    # handle non-serializable np.float32 -> float
    for idx, item in enumerate(predictions):
        for key, value in item.items():
            if isinstance(value, np.float32):
                predictions[idx][key] = float(value)
    return {'response': predictions}

# Run from the command line
# uvicorn src.api.main:app --reload --host=127.0.0.1 --port=8000
