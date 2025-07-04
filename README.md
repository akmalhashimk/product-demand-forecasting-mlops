# 🛍️ Product Demand Forecasting with MLOps

This project implements a scalable, end-to-end machine learning system to forecast daily product sales for multiple retail stores. It covers the entire ML lifecycle including data ingestion, preprocessing, modeling, deployment, monitoring, and retraining.

## 🎯 Project Objective

Accurately forecast daily sales for Rossmann stores using historical sales data, promotions, store characteristics, holidays, and other time-based features. The system is designed to support inventory planning and decision-making at scale.

## 🧠 Key Features

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

## 📊 Dataset

This project uses the [Rossmann Store Sales dataset](https://www.kaggle.com/competitions/rossmann-store-sales/data) from Kaggle.

> 📁 The raw dataset files (`train.csv`, `store.csv`, etc.) are not included in this repository. Please download them manually from Kaggle and place them in the `data/raw/` directory.

## 🚀 Roadmap

- [x] Repository and folder structure setup
- [x] Dataset download and preprocessing
- [x] Exploratory data analysis
- [x] Feature engineering
- [x] Baseline model creation
- [x] Hyperparameter tuning and evaluation
- [x] MLflow experiment tracking
- [x] API deployment with FastAPI
- [x] CI/CD setup with GitHub Actions
- [x] Docker containerization
- [x] Deployment to cloud (GCP or AWS)
- [x] Monitoring and retraining pipeline
- [x] Business dashboard with Streamlit

## 🧑‍💻 How to Run

```bash
# 1. Clone this repository
git clone https://github.com/akmalhashimk/product-demand-forecasting-mlops.git
cd product-demand-forecasting-mlops

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the dataset from Kaggle
and place the CSV files under:
#    data/raw/

# 4. Run training script
python src/train.py

# 5. Start the FastAPI server
cd api/
uvicorn main:app --reload
```
## 📜 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
