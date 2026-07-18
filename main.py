import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# -----------------------------
# Load Dataset
# -----------------------------
data_path = "data.csv"

if not os.path.exists(data_path):
    raise FileNotFoundError(
        "Dataset not found. Please place your dataset as 'data.csv' in the project folder."
    )

df = pd.read_csv(data_path)

print("Dataset loaded successfully!")
print(f"Number of rows: {len(df)}")

# -----------------------------
# Clean Column Names
# -----------------------------
df.columns = df.columns.str.strip()

print("\nColumns:")
print(df.columns.tolist())

# -----------------------------
# Define Target Variable
# -----------------------------
target = "jumlah_guru"

if target not in df.columns:
    raise ValueError(f"Target column '{target}' not found in dataset.")

# Remove target and optional year column
features_df = df.drop(columns=[target, "tahun"], errors="ignore")

# -----------------------------
# Identify Features
# -----------------------------
categorical_features = features_df.select_dtypes(include=["object"]).columns
numerical_features = features_df.select_dtypes(include=np.number).columns

print("\nCategorical Features:")
print(categorical_features.tolist())

print("\nNumerical Features:")
print(numerical_features.tolist())

# -----------------------------
# One-Hot Encoding
# -----------------------------
X = pd.get_dummies(
    features_df,
    columns=categorical_features,
    drop_first=True
)

y = df[target]

print(f"\nFeature Matrix Shape: {X.shape}")
print(X.head())

print(f"\nTarget Shape: {y.shape}")
print(y.head())

# -----------------------------
# Train-Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Linear Regression Model...")

# -----------------------------
# Train Model
# -----------------------------
model = LinearRegression()
model.fit(X_train, y_train)

# -----------------------------
# Prediction
# -----------------------------
print("Making Predictions...")

y_pred = model.predict(X_test)

# -----------------------------
# Evaluation
# -----------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("-" * 40)
print(f"Mean Absolute Error (MAE): {mae:.2f}")
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"R-squared (R²): {r2:.4f}")
