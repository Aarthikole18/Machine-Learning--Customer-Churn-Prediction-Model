import shap
import pandas as pd
import joblib

model = joblib.load("models/churn_model.pkl")

df = pd.read_csv("data/churn.csv")

X = df.drop("churn", axis=1)

explainer = shap.Explainer(model)
shap_values = explainer(X)

print("Top features affecting churn:")
shap.plots.bar(shap_values)