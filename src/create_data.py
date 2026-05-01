import numpy as np
import pandas as pd

np.random.seed(42)

n = 5000

df = pd.DataFrame({
    "tenure": np.random.randint(1, 72, n),
    "monthly_charges": np.random.randint(200, 2000, n),
    "support_tickets": np.random.randint(0, 10, n),
    "usage": np.random.randint(1, 100, n),
})

df["churn"] = (
    (df["support_tickets"] > 5) |
    (df["usage"] < 20) |
    (df["monthly_charges"] > 1500)
).astype(int)

df.to_csv("data/churn.csv", index=False)

print("Dataset created successfully!")