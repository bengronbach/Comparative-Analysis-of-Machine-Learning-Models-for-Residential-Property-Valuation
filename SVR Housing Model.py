# Imports
import os, numpy as np, pandas as pd, kagglehub, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.compose import TransformedTargetRegressor
from scipy.stats import loguniform, uniform

# Load data
path = kagglehub.dataset_download("yasserh/housing-prices-dataset")
df = pd.read_csv(os.path.join(path, "Housing.csv"))
y = df["price"].astype(float)
X = df.drop(columns=["price"])

# Fit Data into Columns
cat_cols = X.select_dtypes(include=["object"]).columns
num_cols = X.select_dtypes(exclude=["object"]).columns

# Preprocess
pre = ColumnTransformer([
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
])

log_t = FunctionTransformer(np.log1p, inverse_func=np.expm1, validate=False)
svr = TransformedTargetRegressor(
    regressor=SVR(kernel="rbf"),
    transformer=log_t   
)

pipe = Pipeline([("preprocessor", pre), ("regressor", svr)])

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Optimize C, gamma, epsilon Values
param_dist = {
    "regressor__regressor__C": loguniform(1e2, 1e5),       # 1e2..1e5
    "regressor__regressor__epsilon": uniform(0.005, 0.15), # 0.005..0.155
    "regressor__regressor__gamma": loguniform(1e-3, 1e0),  # 1e-3..1
}

cv = KFold(n_splits=5, shuffle=True, random_state=42)
rs = RandomizedSearchCV(
    pipe, param_distributions=param_dist, n_iter=40,
    scoring="r2", cv=cv, random_state=42, n_jobs=1, verbose=1,
    return_train_score=True
)

rs.fit(X_train, y_train)
best = rs.best_estimator_
print("Best params:", rs.best_params_)
print("CV R² (mean):", rs.best_score_)

# Evaluate Accuracy
y_pred = best.predict(X_test)
r2  = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse= np.sqrt(mean_squared_error(y_test, y_pred))
print("\nTest Performance")
print(f"R²:   {r2:.3f}")
print(f"MAE:  {mae:,.0f}")
print(f"RMSE: {rmse:,.0f}")

# Plot Results
plt.figure(figsize=(7,6))
plt.scatter(y_test, y_pred, alpha=0.65)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
plt.plot(lims, lims, "r--", linewidth=2, label='Perfect Prediction')
plt.xlim(lims); plt.ylim(lims)
plt.xlabel("Actual Price"); plt.ylabel("Predicted Price")
plt.title("SVR (log-target): Actual vs Predicted Housing Prices")
plt.grid(True); plt.tight_layout(); plt.legend(); plt.show()
