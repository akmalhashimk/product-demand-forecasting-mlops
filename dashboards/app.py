# dashboards/app.py

import streamlit as st
import pandas as pd
import joblib
import os
import datetime

# Title
st.title("📈 Product Demand Forecasting Dashboard")

# Load model
MODEL_PATH = "../models/xgboost_model.pkl"

@st.cache_resource
def load_model():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error("❌ Model not found. Please train it first.")
        return None

model = load_model()

# Upload CSV
st.subheader("📤 Upload CSV file (test data format)")
uploaded_file = st.file_uploader("Choose a file", type=["csv"])

if uploaded_file and model:
    df = pd.read_csv(uploaded_file)
    
    if 'Date' in df.columns:
        # Feature engineering
        df["Date"] = pd.to_datetime(df["Date"])
        df["Year"] = df["Date"].dt.year
        df["Month"] = df["Date"].dt.month
        df["Day"] = df["Date"].dt.day
        df["DayOfWeek"] = df["Date"].dt.dayofweek
        df.drop(columns=["Date", "Customers", "Open"], errors="ignore", inplace=True)

        # Prediction
        preds = model.predict(df)
        df["Predicted Sales"] = preds.round(2)

        st.success("✅ Prediction complete!")
        st.dataframe(df.head())

        # Download
        st.download_button("⬇️ Download Predictions", df.to_csv(index=False), file_name="predictions.csv")

    else:
        st.error("Please include a 'Date' column in your CSV.")
      
