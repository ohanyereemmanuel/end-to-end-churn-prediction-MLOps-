"""
api/main.py — guaranteed feature names fix
"""

import sys
import traceback
from pathlib import Path

import joblib
import pandas as pd
import shap
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

app = FastAPI(title="Churn Prediction API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_DIR     = Path("models")
DATA_DIR      = Path("data/processed")
preprocessor  = None
clf           = None
expected_cols = None
feature_names = None


@app.on_event("startup")
def load_model():
    global preprocessor, clf, expected_cols, feature_names

    try:
        from huggingface_hub import hf_hub_download
        print("Downloading from HuggingFace...")
        prep_path = hf_hub_download(
            repo_id="DocStrange/churn-prediction",
            filename="preprocessor.pkl"
        )
        clf_path = hf_hub_download(
            repo_id="DocStrange/churn-prediction",
            filename="churn_model.pkl"
        )
        preprocessor = joblib.load(prep_path)
        clf          = joblib.load(clf_path)
        print("✅ Models loaded from HuggingFace!")
    except Exception as e:
        print(f"HuggingFace failed ({e}), loading local files...")
        preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
        clf          = joblib.load(MODEL_DIR / "churn_model.pkl")
        print("✅ Models loaded from local files!")
    # Get expected input columns from training data
    x_train_path = hf_hub_download(repo_id="DocStrange/churn-prediction", filename="X_train.csv")
    X_train = pd.read_csv(x_train_path)
    expected_cols = list(X_train.columns)
    print(f"Input columns ({len(expected_cols)}): {expected_cols[:4]}...")

    # Get output feature names by transforming one real row
    sample        = X_train.iloc[[0]]
    X_proc        = preprocessor.transform(sample)
    n_features    = X_proc.shape[1]

    # Try to get real names from ColumnTransformer
    try:
        ct         = preprocessor.named_steps["preprocessor"]
        raw_names  = list(ct.get_feature_names_out())
        clean      = []
        for name in raw_names:
            if "__" in name:
                name = name.split("__")[-1]
            name = name.replace("_", " ").title()
            clean.append(name)
        feature_names = clean
        print(f"✅ Real feature names loaded: {feature_names[:5]}")
    except Exception as e:
        print(f"Warning: could not get feature names ({e})")
        print("Using generic names instead")
        feature_names = [f"Feature {i}" for i in range(n_features)]

    print(f"✅ Model ready — {n_features} output features")


def align_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Match columns to training data exactly."""
    if not expected_cols:
        return df
    for col in expected_cols:
        if col not in df.columns:
            df[col] = "No"
    return df[expected_cols]


def explain(df_raw: pd.DataFrame, top_n: int = 5) -> list[dict]:
    """SHAP explanation with real feature names."""
    df_aligned = align_columns(df_raw.copy())
    X_proc     = preprocessor.transform(df_aligned)
    explainer  = shap.TreeExplainer(clf)
    shap_vals  = explainer.shap_values(X_proc)[0]

    names = feature_names or [f"Feature {i}" for i in range(len(shap_vals))]

    pairs = sorted(
        zip(names, shap_vals),
        key=lambda x: abs(x[1]),
        reverse=True
    )[:top_n]

    return [
        {"feature": str(f), "impact": round(float(v), 4)}
        for f, v in pairs
    ]


class CustomerFeatures(BaseModel):
    tenure:           int   = Field(..., ge=0, le=72)
    monthlycharges:   float = Field(..., gt=0)
    totalcharges:     float = 0.0
    gender:           str   = "Male"
    seniorcitizen:    str   = "No"
    partner:          str   = "No"
    dependents:       str   = "No"
    phoneservice:     str   = "Yes"
    multiplelines:    str   = "No"
    internetservice:  str   = "Fiber optic"
    onlinesecurity:   str   = "No"
    onlinebackup:     str   = "No"
    deviceprotection: str   = "No"
    techsupport:      str   = "No"
    streamingtv:      str   = "No"
    streamingmovies:  str   = "No"
    contract:         str   = "Month-to-month"
    paperlessbilling: str   = "Yes"
    paymentmethod:    str   = "Electronic check"


class PredictionResponse(BaseModel):
    churn_probability: float
    churn_prediction:  bool
    risk_tier:         str
    top_shap_factors:  list[dict]


def get_risk_tier(prob: float) -> str:
    if prob < 0.35: return "LOW"
    if prob < 0.65: return "MEDIUM"
    return "HIGH"


@app.get("/health")
def health():
    return {
        "status":        "healthy",
        "model_loaded":  clf is not None,
        "sample_names":  feature_names[:5] if feature_names else [],
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    if clf is None:
        raise HTTPException(503, "Model not loaded")
    try:
        df_raw     = pd.DataFrame([customer.model_dump()])
        df_aligned = align_columns(df_raw.copy())
        X_proc     = preprocessor.transform(df_aligned)
        prob       = float(clf.predict_proba(X_proc)[0, 1])
        factors    = explain(df_raw, top_n=5)
    except Exception:
        traceback.print_exc()
        raise HTTPException(500, detail=traceback.format_exc()[-800:])

    return PredictionResponse(
        churn_probability = round(prob, 4),
        churn_prediction  = prob >= 0.5,
        risk_tier         = get_risk_tier(prob),
        top_shap_factors  = factors,
    )

    
