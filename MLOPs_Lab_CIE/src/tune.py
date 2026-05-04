import pandas as pd
import mlflow
import random
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingRegressor
import json
import os

df = pd.read_csv("data/training_data.csv")

X = df.drop("irrigation_hours", axis=1)
y = df["irrigation_hours"]

param_grid = {
    "n_estimators": [50, 150],
    "learning_rate": [0.05, 0.1, 0.2],
    "max_depth": [3, 5, 10]
}

trials = 5
best_mae = float("inf")
best_params = None

with mlflow.start_run(run_name="tuning-cropsense"):
    for i in range(trials):
        params = {
            "n_estimators": random.choice(param_grid["n_estimators"]),
            "learning_rate": random.choice(param_grid["learning_rate"]),
            "max_depth": random.choice(param_grid["max_depth"]),
        }

        with mlflow.start_run(nested=True):
            model = GradientBoostingRegressor(**params, random_state=42)

            scores = -cross_val_score(
                model, X, y,
                scoring="neg_mean_absolute_error",
                cv=5
            )

            mae = scores.mean()

            mlflow.log_params(params)
            mlflow.log_metric("mae", mae)

            if mae < best_mae:
                best_mae = mae
                best_params = params

os.makedirs("results", exist_ok=True)

with open("results/step2_s2.json", "w") as f:
    json.dump({
        "search_type": "random",
        "n_folds": 5,
        "total_trials": trials,
        "best_params": best_params,
        "best_mae": best_mae,
        "best_cv_mae": best_mae,
        "parent_run_name": "tuning-cropsense"
    }, f, indent=4)
