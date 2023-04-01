from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI()

MODEL_NAME = 'Gooogr/xlm-roberta-base-pie'

class TextRequest(BaseModel):
    text: str

nlp = pipeline("token-classification", model=MODEL_NAME, tokenizer=MODEL_NAME)

@app.post("/predict_ner")
def predict_ner(text_request: TextRequest):
    predictions = nlp(text_request.text)
    print(predictions)
    return {"predictions": [prediction["word"] for prediction in predictions]}

# uvicorn src.api.main:app --reload --host=127.0.0.1 --port=8000