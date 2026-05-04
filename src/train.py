import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
import json
import os
from math import sqrt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# ----------------------------
# Load data
# ----------------------------
df = pd.read_csv("data/training_data.csv")

X = df.drop("irrigation_hours", axis=1)
y = df["irrigation_hours"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# MLflow setup
# ----------------------------
mlflow.set_experiment("cropsense-irrigation-hours")

results = []
best_model = None
best_mae = float("inf")
best_name = None

models = {
    "SVR": SVR(),
    "GradientBoosting": GradientBoostingRegressor(random_state=42)
}

# ----------------------------
# Train & log models
# ----------------------------
for name, model in models.items():
    with mlflow.start_run(run_name=name):

        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        mae = mean_absolute_error(y_test, preds)
        rmse = sqrt(mean_squared_error(y_test, preds))

        # Log params & metrics
        mlflow.log_params(model.get_params())
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.set_tag("domain", "precision_farming")

        # Log model
        mlflow.sklearn.log_model(model, name=name)

        results.append({
            "name": name,
            "mae": mae,
            "rmse": rmse
        })

        # Track best model
        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_name = name

# ----------------------------
# Retrain best model on FULL dataset
# ----------------------------
if best_name == "SVR":
    final_model = SVR(**best_model.get_params())
else:
    final_model = GradientBoostingRegressor(**best_model.get_params())

final_model.fit(X, y)

# ----------------------------
# Save model
# ----------------------------
os.makedirs("models", exist_ok=True)
joblib.dump(final_model, "models/model.pkl")

# ----------------------------
# Save results JSON
# ----------------------------
os.makedirs("results", exist_ok=True)

with open("results/step1_s1.json", "w") as f:
    json.dump({
        "experiment_name": "cropsense-irrigation-hours",
        "models": results,
        "best_model": best_name,
        "best_metric_name": "mae",
        "best_metric_value": best_mae
    }, f, indent=4)

print("✅ Training complete")
print(f"🏆 Best model: {best_name} (MAE: {best_mae:.4f})")
print("📦 Model saved to models/model.pkl")
