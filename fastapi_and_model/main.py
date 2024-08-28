from fastapi import FastAPI, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.staticfiles import StaticFiles
from layers import PositionalEncoding, TransformerBlock
from fastapi.middleware.cors import CORSMiddleware
import pickle

custom_objects = {'PositionalEncoding': PositionalEncoding, 'TransformerBlock': TransformerBlock}
model = load_model('sentiment_model.keras', custom_objects=custom_objects)
MAX_SEQUENCE_LENGTH = 200
# Initialize the tokenizer 
with open('tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
app = FastAPI()
@app.get("/predict")
async def predict_sentiment(review: str):
    try:
        processed_review = preprocess_review(review, tokenizer, MAX_SEQUENCE_LENGTH)
        prediction = model.predict(processed_review)
        print(f"Prediction: {prediction}")
        sentiment = "positive" if prediction[0] > 0.5 else "negative"
        return {"prediction": sentiment}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
#CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"],  
)
app.mount("/", StaticFiles(directory="static",html = True), name="static")


def preprocess_review(review, tokenizer, max_sequence_length):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen=max_sequence_length)
    
    return padded_sequence


