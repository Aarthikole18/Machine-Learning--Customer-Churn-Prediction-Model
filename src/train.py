import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score
import joblib

# -------------------------
# Load real dataset
# -------------------------
df = pd.read_csv("data/telco_churn.csv")

# -------------------------
# Clean data
# -------------------------
df = df.dropna()

# Convert target
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

# Drop customer ID
df = df.drop("customerID", axis=1)

# -------------------------
# Encode categorical columns
# -------------------------
for col in df.select_dtypes(include="object").columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

# -------------------------
# Split
# -------------------------
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------
# Model
# -------------------------
model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.05,
    eval_metric="logloss"
)

model.fit(X_train, y_train)

# -------------------------
# Evaluation
# -------------------------
pred = model.predict(X_test)
proba = model.predict_proba(X_test)[:,1]

print("\nCLASSIFICATION REPORT\n")
print(classification_report(y_test, pred))

print("\nROC-AUC:", roc_auc_score(y_test, proba))

# -------------------------
# Save model
# -------------------------
joblib.dump(model, "models/churn_model.pkl")
print("\nModel saved successfully!")