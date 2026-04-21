import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import joblib
from sklearn.metrics import (
    roc_auc_score, f1_score, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s — %(levelname)s — %(message)s")
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODEL_DIR     = Path("models")
REPORTS_DIR   = Path("reports")
REPORTS_DIR.mkdir(exist_ok=True)


def load_artifacts():
    preprocessor = joblib.load(MODEL_DIR / "preprocessor.pkl")
    clf          = joblib.load(MODEL_DIR / "churn_model.pkl")
    X_test       = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test       = pd.read_csv(PROCESSED_DIR / "y_test.csv").squeeze()
    return preprocessor, clf, X_test, y_test


def print_report(clf, preprocessor, X_test, y_test):
    X_proc = preprocessor.transform(X_test)
    y_prob = clf.predict_proba(X_proc)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)
    logger.info("\n" + classification_report(
        y_test, y_pred, target_names=["No Churn", "Churn"]
    ))
    logger.info(f"ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")
    return y_prob, y_pred


def plot_roc_curve(clf, preprocessor, X_test, y_test):
    X_proc = preprocessor.transform(X_test)
    fig, ax = plt.subplots(figsize=(7, 5))
    y_prob  = clf.predict_proba(X_proc)[:, 1]
    from sklearn.metrics import RocCurveDisplay
    RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax, name="XGBoost")
    ax.set_title("ROC Curve — Churn Prediction")
    ax.grid(alpha=0.3)
    path = REPORTS_DIR / "roc_curve.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_confusion_matrix(clf, preprocessor, X_test, y_test):
    X_proc = preprocessor.transform(X_test)
    y_pred = clf.predict(X_proc)
    fig, ax = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred,
        display_labels=["No Churn", "Churn"],
        colorbar=False, ax=ax
    )
    ax.set_title("Confusion Matrix")
    path = REPORTS_DIR / "confusion_matrix.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()


def plot_shap_summary(clf, preprocessor, X_test):
    X_proc      = preprocessor.transform(X_test)
    explainer   = shap.TreeExplainer(clf)
    shap_values = explainer.shap_values(X_proc)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(shap_values, X_proc, show=False)
    plt.title("SHAP Summary — Feature Importance")
    plt.tight_layout()
    path = REPORTS_DIR / "shap_summary.png"
    plt.savefig(path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved: {path}")
    plt.close()

def get_individual_explanation(preprocessor, clf, customer_data: pd.DataFrame, top_n: int = 5) -> dict:
    """Returns SHAP explanation with real readable feature names."""
    X_proc    = preprocessor.transform(customer_data)
    explainer = shap.TreeExplainer(clf)
    shap_vals = explainer.shap_values(X_proc)
    shap_row  = shap_vals[0]

    # Get names from the ColumnTransformer step directly
    ct            = preprocessor.named_steps["preprocessor"]
    raw_names     = list(ct.get_feature_names_out())

    # Clean up prefixes like "num__tenure" → "tenure"
    clean_names = []
    for name in raw_names:
        if "__" in name:
            name = name.split("__")[-1]
        # Make names human readable
        name = name.replace("_", " ").title()
        clean_names.append(name)

    # Sort by absolute impact, take top N
    feature_impacts = sorted(
        zip(clean_names, shap_row),
        key=lambda x: abs(x[1]),
        reverse=True,
    )[:top_n]

    return {
        "top_shap_factors": [
            {"feature": f, "impact": round(float(v), 4)}
            for f, v in feature_impacts
        ]
    }
if __name__ == "__main__":
    logger.info("Loading model and data...")
    preprocessor, clf, X_test, y_test = load_artifacts()

    logger.info("Classification report...")
    print_report(clf, preprocessor, X_test, y_test)

    logger.info("Plotting ROC curve...")
    plot_roc_curve(clf, preprocessor, X_test, y_test)

    logger.info("Plotting confusion matrix...")
    plot_confusion_matrix(clf, preprocessor, X_test, y_test)

    logger.info("Computing SHAP values (takes 1-2 mins)...")
    plot_shap_summary(clf, preprocessor, X_test)

    logger.info("✅ Done! Check your reports/ folder")