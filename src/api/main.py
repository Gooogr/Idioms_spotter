from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
import numpy as np
import uvicorn

app = FastAPI()

MODEL_NAME = 'Gooogr/xlm-roberta-base-pie'

class TextRequest(BaseModel):
    text: str

# TODO: add device selection from created src/model/utils

nlp = pipeline("token-classification", model=MODEL_NAME, tokenizer=MODEL_NAME)

@app.post("/predict_ner")
def predict_ner(text_request: TextRequest):
    predictions = nlp(text_request.text)
    
    # handle non-serializable np.float32 -> float
    for idx, item in enumerate(predictions):
        for key, value in  item.items():
            if isinstance(value, np.float32):
                predictions[idx][key] = float(value)
    return {'response': predictions}

# Run from the command line
# uvicorn src.api.main:app --reload --host=127.0.0.1 --port=8000
    
