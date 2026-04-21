# 🔁 Customer Churn Prediction — End-to-End ML System

> Predicting which telecom customers will churn using a production-grade ML pipeline with experiment tracking, explainability, and a live REST API.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![MLflow](https://img.shields.io/badge/MLflow-tracked-orange?logo=mlflow)
![FastAPI](https://img.shields.io/badge/FastAPI-deployed-green?logo=fastapi)
![Docker](https://img.shields.io/badge/Docker-containerised-blue?logo=docker)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## 📌 Problem Statement

A telecom company loses ~$1,500 per churned customer in acquisition costs to replace them. Identifying at-risk customers **before** they leave — and intervening with retention offers — is far cheaper. This project builds a binary classifier that flags customers likely to churn in the next 30 days, with business-interpretable explanations for every prediction.

---

## 🏆 Results

| Model | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression (baseline) | 0.791 | 0.63 | 0.57 | 0.60 |
| Random Forest | 0.851 | 0.71 | 0.68 | 0.69 |
| **XGBoost (final)** | **0.887** | **0.79** | **0.74** | **0.76** |
| XGBoost + feature selection | 0.883 | 0.80 | 0.73 | 0.76 |

> **Business impact**: At 0.74 recall, the model catches ~74% of churners. Assuming a 10% churn rate across 50,000 customers and a $300 retention offer cost with 40% success rate, this translates to ~$2.1M in recovered annual revenue.

---

## 🗂️ Dataset

- **Source**: [Telco Customer Churn — IBM Sample Dataset (Kaggle)](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Size**: 7,043 customers × 21 features
- **Target**: `Churn` — binary (Yes/No) — **26.5% positive rate** (imbalanced)
- **Key features**: tenure, contract type, monthly charges, internet service, tech support

---

## 🧠 Key Concepts Demonstrated

- **EDA & class imbalance handling** — SMOTE oversampling + class weights
- **Feature engineering** — interaction terms, tenure buckets, charge ratios
- **Experiment tracking** — MLflow: every run logged with params, metrics, and artefacts
- **Model selection** — nested cross-validation, Optuna hyperparameter tuning
- **Explainability** — SHAP waterfall plots, global feature importance, what-if analysis
- **Deployment** — FastAPI REST endpoint, Dockerised, ready for cloud deployment
- **Testing** — pytest unit tests for feature pipeline

---

## 🚀 Quickstart

### 1. Clone and install

```bash
git clone https://github.com/yourusername/churn-prediction.git
cd churn-prediction
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Download data

```bash
# Place the Kaggle CSV at:
data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv

# Or use the Kaggle API:
kaggle datasets download -d blastchar/telco-customer-churn -p data/raw/ --unzip
```

### 3. Run the full pipeline

```bash
# Train and log to MLflow
python src/train.py

# View experiment results
mlflow ui   # open http://localhost:5000
```

### 4. Start the API

```bash
uvicorn api.main:app --reload
# Docs at http://localhost:8000/docs
```

### 5. Docker (production)

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

---

## 📊 Explainability

Every prediction comes with a SHAP explanation. Example output for a high-risk customer:

```
Churn probability: 0.84 (HIGH RISK)

Top factors pushing toward churn:
  + Month-to-month contract     (+0.21)
  + High monthly charges        (+0.18)
  + No tech support             (+0.14)
  + Short tenure (3 months)     (+0.11)

Top factors pushing against churn:
  - Has phone service           (-0.06)
  - Uses online backup          (-0.04)
```

---

## 🗺️ Architecture

```
Raw CSV → DataPipeline → FeatureEngineering → XGBoostClassifier
                                                      ↓
                                              MLflow (tracking)
                                                      ↓
                                           SHAP Explainer → Reports
                                                      ↓
                                            FastAPI (REST API)
                                                      ↓
                                           Docker Container → Cloud
```

---

## 📁 Project Structure

```
churn-prediction/
├── data/raw/                  # raw, immutable data
├── data/processed/            # cleaned, featurised data
├── notebooks/01_EDA.ipynb     # exploratory analysis
├── src/
│   ├── data_pipeline.py       # ingestion and cleaning
│   ├── features.py            # feature engineering
│   ├── train.py               # training + MLflow logging
│   ├── evaluate.py            # metrics + SHAP plots
│   └── predict.py             # inference
├── api/main.py                # FastAPI REST service
├── tests/test_features.py     # unit tests
├── Dockerfile
├── requirements.txt
└── config.yaml
```

---

## 🔬 MLflow Experiment Tracking

All training runs are logged automatically:

```python
mlflow.log_param("max_depth", 6)
mlflow.log_metric("roc_auc", 0.887)
mlflow.log_artifact("shap_summary.png")
mlflow.sklearn.log_model(model, "xgboost_churn_model")
```

Open `mlflow ui` to compare runs, view metrics over time, and load any registered model.

---

## 📡 API Reference

**POST** `/predict`

```json
{
  "tenure": 3,
  "contract": "Month-to-month",
  "monthly_charges": 85.5,
  "internet_service": "Fiber optic",
  "tech_support": "No",
  "online_security": "No",
  "payment_method": "Electronic check"
}
```

**Response:**
```json
{
  "churn_probability": 0.84,
  "churn_prediction": true,
  "risk_tier": "HIGH",
  "top_shap_factors": [
    {"feature": "contract_month-to-month", "impact": 0.21},
    {"feature": "monthly_charges", "impact": 0.18}
  ]
}
```

---

## 🧪 Testing

```bash
pytest tests/ -v
```

---

## 📦 Requirements

See `requirements.txt`. Key libraries:
- `xgboost`, `scikit-learn`, `imbalanced-learn`
- `mlflow`, `optuna`
- `shap`
- `fastapi`, `uvicorn`, `pydantic`
- `pandas`, `numpy`

---

## 📄 License

MIT — use freely, attribution appreciated.

---

*Built to demonstrate production ML engineering. Questions? Open an issue.*
