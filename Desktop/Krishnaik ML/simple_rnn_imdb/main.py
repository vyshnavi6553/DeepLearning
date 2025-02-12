# Step 1: Import Libraries and Load the Model
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from fastapi import FastAPI
import pandas as pd
import pickle
from pydantic import BaseModel
import uvicorn,json
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}


app=FastAPI()
# Load the pre-trained model with ReLU activation
model = load_model('simple_rnn_imdb_model.h5')



word_index = {}  

# Function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]  
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)  # Ensure proper input shape
    return padded_review

# Define request model
class ReviewRequest(BaseModel):
    review: str  

# API endpoint for predictions
@app.post("/review")
def predict_review(data: ReviewRequest):
    input_review = preprocess_text(data.review)  # Process text
    prediction = model.predict(input_review)  # Get model prediction
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'  # Convert to label

    # Return response in proper JSON format
    return {"sentiment": sentiment, "prediction_score": float(prediction[0][0])}


if __name__ == '__main__':
    uvicorn.run(app, port=5000)
