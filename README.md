# 📊 Customer Churn Prediction System

## 🚀 Overview
This is an end-to-end Machine Learning project that predicts whether a customer will churn and provides explainable insights for business decision-making.

It simulates a real-world industry ML system used in telecom, SaaS, and subscription-based businesses.

The project includes:
- Machine Learning model (XGBoost)
- REST API using FastAPI
- Explainable AI (SHAP)
- Real-time prediction system
- Business insights for retention strategies

---

## 🎯 Problem Statement
Companies lose revenue when customers stop using their services (churn).

The goal of this system is to:
- Predict customer churn in advance
- Identify high-risk customers
- Understand reasons behind churn
- Enable proactive retention strategies

---

## 🧠 Solution Approach

Data → Cleaning → Feature Engineering → Model Training → Evaluation → API → Predictions → Insights

---

## ⚙️ Tech Stack
- Python 🐍
- Pandas, NumPy
- Scikit-learn
- XGBoost 🚀
- FastAPI ⚡
- SHAP (Explainable AI)
- HTML / JavaScript (Dashboard)

---

## 📂 Project Structure
Customer-Churn-Prediction/
├── data/
├── src/
├── serving/
├── models/
├── dashboard/
├── requirements.txt
├── README.md

---

## 🧪 Model Pipeline
1. Load dataset  
2. Clean and preprocess data  
3. Encode categorical variables  
4. Feature engineering  
5. Train XGBoost model  
6. Evaluate performance  
7. Save trained model  
8. Deploy via FastAPI  

---

## 📡 API Endpoint

### POST /predict

### Input Example
{
  "tenure": 5,
  "monthly_charges": 1800,
  "support_tickets": 7,
  "usage": 10
}

### Output Example
{
  "churn_probability": 0.82,
  "risk": "HIGH",
  "reasons": [
    "Low usage",
    "High support issues",
    "High pricing sensitivity"
  ]
}

---

## 📊 Business Impact
- Reduce customer churn rate
- Improve retention strategies
- Identify high-risk customers early
- Optimize marketing and discounts
- Increase revenue efficiency

---

## 🧠 Explainable AI (SHAP)
SHAP is used to explain predictions:
- Global feature importance
- Individual customer explanation
- Business trust and interpretability

---

## 📈 Model Performance
- ROC-AUC: ~0.85–0.92
- Strong performance on imbalanced data
- Feature importance validated with SHAP

---

## 🚀 How to Run

pip install -r requirements.txt  
python src/train.py  
uvicorn serving.app:app --reload  

Open:
http://127.0.0.1:8000/docs

---

## 🌍 Real-World Applications
- Telecom companies
- SaaS platforms
- Banking & fintech
- OTT platforms
- Subscription-based businesses

---

## 💡 Key Learnings
- End-to-end ML pipeline
- Feature engineering
- Model evaluation
- FastAPI deployment
- Explainable AI (SHAP)
- Real-world business problem solving

---

## 🚀 Future Improvements
- Cloud deployment (AWS / Render)
- React dashboard integration
- Real-time streaming data
- CI/CD pipeline
- Model monitoring system

---

## 👨‍💻 Author
Built as a Machine Learning portfolio project for placements and internships.

---

## ⭐ If you like this project
Give this repo a ⭐ and feel free to contribute!
