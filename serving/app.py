from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
import os

app = FastAPI(title="Customer Churn API")

# -----------------------------
# FIX: absolute-safe model path
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

model = joblib.load(MODEL_PATH)

# -----------------------------
# Root route (Render health check)
# -----------------------------
@app.get("/")
def home():
    return {"message": "Churn Prediction API is live 🚀"}

# -----------------------------
# Input schema
# -----------------------------
class Customer(BaseModel):
    tenure: float
    monthly_charges: float
    support_tickets: float
    usage: float

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(customer: Customer):

    input_data = np.array([[
        customer.tenure,
        customer.monthly_charges,
        customer.support_tickets,
        customer.usage
    ]])

    prob = model.predict_proba(input_data)[0][1]

    risk = (
        "HIGH" if prob >= 0.7 else
        "MEDIUM" if prob >= 0.4 else
        "LOW"
    )

    reasons = []

    if customer.usage < 20:
        reasons.append("Low usage")
    if customer.support_tickets > 5:
        reasons.append("High support issues")
    if customer.monthly_charges > 1500:
        reasons.append("High pricing sensitivity")
    if customer.tenure < 6:
        reasons.append("New customer")

    return {
        "churn_probability": float(prob),
        "risk": risk,
        "reasons": reasons
    }