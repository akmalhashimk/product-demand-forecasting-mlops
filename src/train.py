# src/train.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
import mlflow
import mlflow.xgboost
import optuna

# Paths
RAW_TRAIN_PATH = "data/raw/train.csv"
STORE_PATH = "data/raw/store.csv"
MODEL_PATH = "models/xgboost_model.pkl"

def load_and_merge_data():
    train = pd.read_csv(RAW_TRAIN_PATH)
    store = pd.read_csv(STORE_PATH)
    data = pd.merge(train, store, on="Store", how="left")
    data['Date'] = pd.to_datetime(data['Date'])
    return data

def preprocess(data):
    data = data[data["Open"] != 0]
    data.fillna(0, inplace=True)

    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["DayOfWeek"] = data["Date"].dt.dayofweek

    drop_cols = ["Date", "Customers", "Open"]
    data = data.drop(columns=drop_cols, errors='ignore')
    return data

def train_model(data):
    X = data.drop(columns=["Sales"])
    y = data["Sales"]

    def objective(trial):
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        return rmse

    # Run optimization
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=30)

    best_params = study.best_params
    print("✅ Best Params:", best_params)

    # Final model with best params
    with mlflow.start_run():
        mlflow.log_params(best_params)

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        mlflow.log_metric("rmse", rmse)

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.xgboost.log_model(model, artifact_path="model")

        print(f"📊 Final RMSE: {rmse:.2f}")

if __name__ == "__main__":
    data = load_and_merge_data()
    data = preprocess(data)
    train_model(data)
    
