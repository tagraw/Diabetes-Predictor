from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Load the trained model
model = joblib.load("model/diabetes_model.pkl")

app = FastAPI()

class DiabetesInput(BaseModel):
    age: float
    sex: float
    bmi: float
    bp: float
    s1: float
    s2: float
    s3: float
    s4: float
    s5: float
    s6: float


@app.post("/predict")
def predict_diabetes(data: DiabetesInput):
    features = np.array([[data.age, data.sex, data.bmi, data.bp, data.s1, data.s2, data.s3, data.s4, data.s5, data.s6]])
    prediction = model.predict(features)
    return {"predicted_diabetes_progression": float(prediction[0])}