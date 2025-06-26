# src/predict.py

import pandas as pd
import joblib
import os

MODEL_PATH = "models/xgboost_model.pkl"

def preprocess_input(data: pd.DataFrame) -> pd.DataFrame:
    # Extract features from Date
    data["Date"] = pd.to_datetime(data["Date"])
    data["Year"] = data["Date"].dt.year
    data["Month"] = data["Date"].dt.month
    data["Day"] = data["Date"].dt.day
    data["DayOfWeek"] = data["Date"].dt.dayofweek

    # Drop unused columns
    drop_cols = ["Date", "Customers", "Open"]
    data = data.drop(columns=drop_cols, errors='ignore')

    return data

def predict_from_file(input_path):
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Trained model not found. Please run train.py first.")

    model = joblib.load(MODEL_PATH)
    data = pd.read_csv(input_path)
    data_processed = preprocess_input(data)

    predictions = model.predict(data_processed)
    return predictions

if __name__ == "__main__":
    input_file = "data/raw/test.csv"  # you can change this to any new data file
    preds = predict_from_file(input_file)

    # Save to CSV
    output = pd.DataFrame({"Prediction": preds})
    output.to_csv("predictions.csv", index=False)
    print("✅ Predictions saved to predictions.csv")
  
