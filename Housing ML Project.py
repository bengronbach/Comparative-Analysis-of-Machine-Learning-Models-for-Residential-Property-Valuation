# Imports

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import kagglehub

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load data
path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
csv_path = os.path.join(path, "Housing.csv")
df = pd.read_csv(csv_path)

# Define target values
target = "price"
y = df[target]
X = df.drop(columns=[target])

# Sort Columns
numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

# Preprocess Data
preprocess = ColumnTransformer(
    transformers=[
        ("num", "passthrough", numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ],
    remainder="drop",
)

# Train/test split 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.30, random_state=42
)

# Train model
reg = DecisionTreeRegressor(max_depth=5, random_state=42)
pipe = Pipeline([("prep", preprocess), ("model", reg)])
pipe.fit(X_train, y_train)

# Evaluate Accuracy metrics
pred = pipe.predict(X_test)
r2 = r2_score(y_test, pred)
mae = mean_absolute_error(y_test, pred)
rmse = np.sqrt(mean_squared_error(y_test, pred))

print("----- Decision Tree Regressor (Test Set) -----")
print(f"R²:   {r2:.3f}")
print(f"MAE:  {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")

# Plot Decision Tree visual
model = pipe.named_steps["model"]
ohe = pipe.named_steps["prep"].named_transformers_["cat"]

if len(categorical_cols):
    cat_names = ohe.get_feature_names_out(categorical_cols)
else:
    cat_names = np.array([])

feature_names = np.concatenate([numeric_cols, cat_names])

plt.figure(figsize=(18, 10))
plot_tree(
    model,
    feature_names=feature_names,
    filled=True,
    rounded=True,
    fontsize=8
)
plt.title("Decision Tree Regressor (max_depth=4)")
plt.tight_layout()

# Plot Predicted Prices vs Target Prices
plt.figure(figsize=(7, 7))
plt.scatter(y_test, pred, alpha=0.6, edgecolor="k")
plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    "r--", lw=2, label="Perfect Prediction"
)
plt.title("Predicted vs Actual House Prices")
plt.xlabel("Actual Price (In Dollars*10^6)")
plt.ylabel("Predicted Price (In Dollars*10^6)")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()


plt.show()

