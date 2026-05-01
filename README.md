# 📊 Customer Churn Prediction System

## 🚀 Overview
Machine Learning project that predicts customer churn and provides actionable insights using FastAPI and XGBoost.

## 🧠 Features
- Churn prediction using ML model
- Real-time API with FastAPI
- Explainable AI insights
- Risk classification (Low / Medium / High)

## ⚙️ Tech Stack
Python, Pandas, Scikit-learn, XGBoost, FastAPI, SHAP

## 🚀 How to Run
pip install -r requirements.txt  
python src/train.py  
uvicorn serving.app:app --reload  

## 📡 API Endpoint
POST /predict

## 📊 Output
- churn_probability
- risk level
- reasons for churn

## 👨‍💻 Author
Student ML Project for portfolio building