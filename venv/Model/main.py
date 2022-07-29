import pickle
from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from tfidf_vectorizer import Tdidf_vectorizer


with open('model_pickle','rb') as f:
    model = pickle.load(f)

class Msg(BaseModel):
    message : str

app = FastAPI()

@app.get("/")
async def root():
    return {"Sentiment Analysis API setup successful!"}

@app.post("/predict")
async def predict_sentiment(data : Msg):
    data = data.dict()
    message = data['message']
    vector = Tdidf_vectorizer(message)
    vectorized_message = vector.vectorizer()
    sentiment = model.predict(vectorized_message)[0]
    if sentiment == 1:
        return {"sentiment" : "positive"}
    else:
        return {"sentiment" : "negative"}

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)