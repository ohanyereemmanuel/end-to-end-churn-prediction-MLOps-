import logging
import joblib
import os
import sys
from pathlib import Path


# This line fixes the "no module named features" error on Windows
sys.path.insert(0, str(Path(__file__).parent))

import pandas as pd
import mlflow
import optuna
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

from features import build_preprocessor

import mlflow
import mlflow.sklearn
import optuna

# Add this line right here
mlflow.set_tracking_uri("sqlite:///mlflow.db")
os.environ["GIT_PYTHON_REFRESH"] = "quiet"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s"
)
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("models")
MODEL_DIR.mkdir(exist_ok=True)
EXPERIMENT_NAME = "churn-prediction"


def load_splits():
    X_train = pd.read_csv(PROCESSED_DIR / "X_train.csv")
    X_val   = pd.read_csv(PROCESSED_DIR / "X_val.csv")
    X_test  = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_train = pd.read_csv(PROCESSED_DIR / "y_train.csv").squeeze()
    y_val   = pd.read_csv(PROCESSED_DIR / "y_val.csv").squeeze()
    y_test  = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return X_train, X_val, X_test, y_train, y_val, y_test


def preprocess_and_resample(X_train, y_train, X_eval):
    preprocessor   = build_preprocessor()
    X_train_proc   = preprocessor.fit_transform(X_train, y_train)
    X_eval_proc    = preprocessor.transform(X_eval)
    smote          = SMOTE(random_state=42)
    X_res, y_res   = smote.fit_resample(X_train_proc, y_train)
    logger.info(f"After SMOTE — {len(X_res):,} training samples")
    return preprocessor, X_res, y_res, X_eval_proc


def train_baseline(X_train, y_train, X_val, y_val):
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="baseline_logreg"):
        prep, X_tr, y_tr, X_v = preprocess_and_resample(X_train, y_train, X_val)
        clf = LogisticRegression(max_iter=1000, random_state=42)
        clf.fit(X_tr, y_tr)
        y_prob = clf.predict_proba(X_v)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = {
            "roc_auc":   round(roc_auc_score(y_val, y_prob), 4),
            "f1":        round(f1_score(y_val, y_pred), 4),
            "precision": round(precision_score(y_val, y_pred), 4),
            "recall":    round(recall_score(y_val, y_pred), 4),
        }
        mlflow.log_params({"model": "LogisticRegression"})
        mlflow.log_metrics(metrics)
        logger.info(f"Baseline — ROC-AUC: {metrics['roc_auc']} | F1: {metrics['f1']}")
    return metrics


def objective(trial, X_tr, y_tr, X_v, y_val):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 300),
        "max_depth":        trial.suggest_int("max_depth", 3, 7),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    clf = XGBClassifier(
        **params, eval_metric="logloss",
        random_state=42, n_jobs=-1, verbosity=0
    )
    clf.fit(X_tr, y_tr)
    return roc_auc_score(y_val, clf.predict_proba(X_v)[:, 1])


def tune_xgboost(X_tr, y_tr, X_v, y_val, n_trials=30):
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42)
    )
    study.optimize(
        lambda trial: objective(trial, X_tr, y_tr, X_v, y_val),
        n_trials=n_trials,
        show_progress_bar=True,
    )
    logger.info(f"Best ROC-AUC: {study.best_value:.4f}")
    logger.info(f"Best params:  {study.best_params}")
    return study.best_params


def train_final_model(X_train, y_train, X_val, y_val,
                      X_test, y_test, best_params):
    mlflow.set_experiment(EXPERIMENT_NAME)
    with mlflow.start_run(run_name="xgboost_final") as run:
        X_full = pd.concat([X_train, X_val], ignore_index=True)
        y_full = pd.concat([y_train, y_val], ignore_index=True)

        prep, X_res, y_res, X_test_proc = preprocess_and_resample(
            X_full, y_full, X_test
        )
        clf = XGBClassifier(
            **best_params, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0
        )
        clf.fit(X_res, y_res)

        y_prob = clf.predict_proba(X_test_proc)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = {
            "test_roc_auc":   round(roc_auc_score(y_test, y_prob), 4),
            "test_f1":        round(f1_score(y_test, y_pred), 4),
            "test_precision": round(precision_score(y_test, y_pred), 4),
            "test_recall":    round(recall_score(y_test, y_pred), 4),
        }
        mlflow.log_params({"model": "XGBoost", **best_params})
        mlflow.log_metrics(metrics)

        joblib.dump(prep, MODEL_DIR / "preprocessor.pkl")
        joblib.dump(clf,  MODEL_DIR / "classifier.pkl")

        logger.info(f"✅ Model saved! Run ID: {run.info.run_id}")
        for k, v in metrics.items():
            logger.info(f"   {k}: {v}")

    return prep, clf


if __name__ == "__main__":
    logger.info("Step 1/4 — Loading data...")
    X_train, X_val, X_test, y_train, y_val, y_test = load_splits()

    logger.info("Step 2/4 — Training baseline...")
    train_baseline(X_train, y_train, X_val, y_val)

    logger.info("Step 3/4 — Tuning XGBoost (30 trials, ~5 mins)...")
    _, X_tr, y_tr, X_v = preprocess_and_resample(X_train, y_train, X_val)
    best_params = tune_xgboost(X_tr, y_tr, X_v, y_val, n_trials=30)

    logger.info("Step 4/4 — Training final model...")
    train_final_model(X_train, y_train, X_val, y_val,
                      X_test, y_test, best_params)

    logger.info("🎉 All done! Now run: mlflow ui")