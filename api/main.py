# api/main.py

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib
import uvicorn
import os

app = FastAPI(title="Product Demand Forecasting API")

MODEL_PATH = "models/xgboost_model.pkl"

# Load model
model = joblib.load(MODEL_PATH)

# Input schema
class SaleRecord(BaseModel):
    Store: int
    DayOfWeek: int
    Promo: int
    SchoolHoliday: int
    StoreType: str
    Assortment: str
    CompetitionDistance: float
    Promo2: int
    Date: str

# Preprocess input
def preprocess(record: SaleRecord):
    df = pd.DataFrame([record.dict()])
    df["Date"] = pd.to_datetime(df["Date"])
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    df["DayOfWeek"] = df["Date"].dt.dayofweek

    drop_cols = ["Date"]
    df.drop(columns=drop_cols, inplace=True, errors="ignore")
    return df

# Root route
@app.get("/")
def read_root():
    return {"message": "Product Demand Forecasting API is live."}

# Predict route
@app.post("/predict")
def predict_sale(record: SaleRecord):
    data = preprocess(record)
    prediction = model.predict(data)[0]
    return {"predicted_sales": round(prediction, 2)}
  
