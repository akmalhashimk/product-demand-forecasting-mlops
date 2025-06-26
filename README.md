# 🛍️ Product Demand Forecasting with MLOps

This project implements a scalable, end-to-end machine learning system to forecast daily product sales for multiple retail stores. It covers the entire ML lifecycle including data ingestion, preprocessing, modeling, deployment, monitoring, and retraining.

## 🎯 Objective

Accurately forecast daily sales for Rossmann stores using historical sales data, promotions, store characteristics, holidays, and other time-based features. The system is designed to support inventory planning and decision-making at scale.

## 🧠 Features

- End-to-end pipeline for time-series forecasting
- Feature engineering with temporal, promotional, and store-level data
- Models: XGBoost, LightGBM, TimeSeriesSplit
- Experiment tracking with MLflow
- REST API deployment using FastAPI
- CI/CD with GitHub Actions
- Containerization with Docker
- Cloud deployment (GCP/AWS)
- Monitoring with drift detection and retraining pipeline
- Interactive dashboard using Streamlit

- ## 🔧 Tech Stack

| Layer              | Tools Used                                |
|-------------------|--------------------------------------------|
| Data Processing    | Pandas, NumPy, Airflow                     |
| Modeling           | Scikit-learn, XGBoost, LightGBM            |
| Experimentation    | MLflow, Optuna, TimeSeriesSplit            |
| API & Deployment   | FastAPI, Uvicorn, Docker, GCP / AWS        |
| Workflow & CI/CD   | Airflow, GitHub Actions                    |
| Monitoring         | Prometheus, Grafana                        |
| Dashboard          | Streamlit                                  |
