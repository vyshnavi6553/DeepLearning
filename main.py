from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import pandas as pd
import pickle
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI app
app = FastAPI()

# Load the trained model
model = tf.keras.models.load_model('mymodel.h5')

# Load the encoders and scaler
with open('label_encoder_gender1.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo1.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler1.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Define input data model
class CustomerData(BaseModel):
    geography: str
    gender: str
    age: int
    balance: float
    credit_score: int
    estimated_salary: float
    tenure: int
    num_of_products: int
    has_cr_card: int
    is_active_member: int

@app.post("/predict")
def predict_churn(data: CustomerData):
    # Prepare the input data
    input_data = pd.DataFrame({
        'CreditScore': [data.credit_score],
        'Gender': [label_encoder_gender.transform([data.gender])[0]],
        'Age': [data.age],
        'Tenure': [data.tenure],
        'Balance': [data.balance],
        'NumOfProducts': [data.num_of_products],
        'HasCrCard': [data.has_cr_card],
        'IsActiveMember': [data.is_active_member],
        'EstimatedSalary': [data.estimated_salary]
    })

    # One-hot encode 'Geography'
    geo_encoded = onehot_encoder_geo.transform([[data.geography]]).toarray()
    geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
    
    # Combine one-hot encoded columns with input data
    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

    # Scale the input data
    input_data_scaled = scaler.transform(input_data)

    # Predict churn
    prediction = model.predict(input_data_scaled)
    prediction_proba = float(prediction[0][0])

    return {
        "churn_probability": round(prediction_proba, 2),
        "churn_risk": "High" if prediction_proba > 0.5 else "Low"
    }

# Run the app using: uvicorn filename:app --reload

if __name__ == '__main__':
    uvicorn.run(app, port=5000)