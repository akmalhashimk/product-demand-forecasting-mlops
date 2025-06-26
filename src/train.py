# src/train.py

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os
import mlflow
import mlflow.xgboost

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

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    with mlflow.start_run():
        params = {"n_estimators": 100, "learning_rate": 0.1}
        mlflow.log_params(params)

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = mean_squared_error(y_val, preds, squared=False)
        mlflow.log_metric("rmse", rmse)

        # Save model
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, MODEL_PATH)
        mlflow.xgboost.log_model(model, artifact_path="model")

        print(f"✅ Model trained. RMSE: {rmse:.2f}")
        print("📦 Model logged to MLflow.")

if __name__ == "__main__":
    data = load_and_merge_data()
    data = preprocess(data)
    train_model(data)
    
