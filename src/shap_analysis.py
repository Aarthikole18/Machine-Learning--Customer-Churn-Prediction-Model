import pandas as pd
import shap
import joblib
import matplotlib.pyplot as plt

# Load model
model = joblib.load("models/churn_model.pkl")

# Load dataset
df = pd.read_csv("data/telco_churn.csv")

df = df.dropna()
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
df = df.drop("customerID", axis=1)

X = df.drop("Churn", axis=1)

# SHAP explainer
explainer = shap.Explainer(model)
shap_values = explainer(X)

# -------------------------
# GLOBAL IMPORTANCE PLOT
# -------------------------
shap.plots.bar(shap_values)
plt.title("Feature Importance for Customer Churn")
plt.show()

# -------------------------
# SINGLE CUSTOMER EXPLANATION
# -------------------------
print("\nExample explanation for first customer:")
shap.plots.waterfall(shap_values[0])