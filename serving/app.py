from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
import os

# -----------------------------
# Load model safely (IMPORTANT for deployment)
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

model = joblib.load(MODEL_PATH)

# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Customer Churn Prediction API")

# -----------------------------
# CORS (allow frontend access)
# -----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Input schema
# -----------------------------
class Customer(BaseModel):
    tenure: int
    monthly_charges: int
    support_tickets: int
    usage: int

# -----------------------------
# Home route
# -----------------------------
@app.get("/")
def home():
    return {"message": "Customer Churn API is running 🚀"}

# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict")
def predict(customer: Customer):

    # convert input → model format
    data = np.array([[
        customer.tenure,
        customer.monthly_charges,
        customer.support_tickets,
        customer.usage
    ]])

    # prediction probability
    prob = model.predict_proba(data)[0][1]

    # risk classification
    if prob > 0.6:
        risk = "HIGH"
    elif prob > 0.3:
        risk = "MEDIUM"
    else:
        risk = "LOW"

    # simple explainability (business logic)
    reasons = []

    if customer.usage < 20:
        reasons.append("Low usage")
    if customer.support_tickets > 5:
        reasons.append("High support issues")
    if customer.monthly_charges > 1500:
        reasons.append("High pricing sensitivity")
    if customer.tenure < 10:
        reasons.append("New customer risk")

    return {
        "churn_probability": float(prob),
        "risk": risk,
        "reasons": reasons
    }