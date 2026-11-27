from fastapi import FastAPI
from pydantic import BaseModel

from src.serve import load_model, translate


class TranslateRequest(BaseModel):
    text: str


class TranslateResponse(BaseModel):
    translation: str
    time_taken: float


app = FastAPI(title="German-to-English Translation API")

# Module level on purpose: runs once when the server process starts, not per request.
model, tokenizer, config = load_model()


@app.get("/")
def root():
    return {"message": "German-to-English Translation API", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/translate", response_model=TranslateResponse)
def translate_endpoint(request: TranslateRequest):
    result = translate(request.text, model, tokenizer, config)
    return TranslateResponse(**result)
