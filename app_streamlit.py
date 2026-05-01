import streamlit as st
import joblib
import numpy as np
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

model = joblib.load(MODEL_PATH)

st.title("📊 Customer Churn Prediction System")
st.write("Enter customer details to predict churn probability")

# Inputs
tenure = st.number_input("Tenure (months)", 0, 100, 5)
monthly_charges = st.number_input("Monthly Charges", 0, 5000, 1000)
support_tickets = st.number_input("Support Tickets", 0, 20, 2)
usage = st.number_input("Usage Score", 0, 100, 50)

# Predict button
if st.button("Predict Churn"):
    
    input_data = np.array([[tenure, monthly_charges, support_tickets, usage]])
    
    prob = model.predict_proba(input_data)[0][1]

    if prob > 0.7:
        risk = "🔴 HIGH RISK"
    elif prob > 0.4:
        risk = "🟡 MEDIUM RISK"
    else:
        risk = "🟢 LOW RISK"

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {round(prob, 2)}")
    st.write(f"Risk Level: {risk}")

    # Simple reasoning
    st.subheader("Possible Reasons")
    if usage < 20:
        st.write("• Low usage detected")
    if support_tickets > 5:
        st.write("• High support complaints")
    if monthly_charges > 1500:
        st.write("• High pricing sensitivity")
    if tenure < 6:
        st.write("• New customer (low tenure)")