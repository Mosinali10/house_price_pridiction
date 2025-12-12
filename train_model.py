# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Ensure model folder exists
os.makedirs("model", exist_ok=True)

# Load Dataset
# Use whitespace as separator, no header, and provide column names
column_names = [
    "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", 
    "PTRATIO", "B", "LSTAT", "MEDV"
]
df = pd.read_csv("data/housing.csv", sep=r'\s+', names=column_names)

# We will use ONLY 4 features to match the app UI
FEATURES = ["RM", "AGE", "TAX", "LSTAT"]

X = df[FEATURES]
y = df["MEDV"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate model
preds = model.predict(X_test)
mse = mean_squared_error(y_test, preds)
r2 = r2_score(y_test, preds)

print(f"\nModel Training Complete")
print(f"MSE: {mse}")
print(f"R² Score: {r2}")

# Save model
model_path = "model/model.pkl"
joblib.dump(model, model_path)

print(f"Model saved to {model_path}")
