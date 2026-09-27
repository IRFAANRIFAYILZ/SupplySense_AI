"""
train.py
--------
Trains a REAL machine learning model that predicts whether an order is
likely to be delayed (Option A: Delivery Delay Prediction).

Model: RandomForestClassifier (scikit-learn)
Target: is_delayed (1 = order was Delayed, 0 = order was On Time)
        (rows with status == "In Transit" are excluded — outcome not yet known)

Features:
  - supplier_on_time_rate       (historical, from suppliers.csv)
  - supplier_avg_delivery_days  (historical, from suppliers.csv)
  - supplier_quality_score
  - supplier_risk_score
  - category                    (one-hot encoded, from inventory/product category)
  - quantity
  - priority                    (one-hot encoded)
  - order_month                 (captures seasonality)

This script:
  1. Loads and joins suppliers.csv, orders.csv, inventory.csv
  2. Builds a leakage-safe feature set (no feature directly derived from the
     row's own outcome)
  3. Splits into train/test
  4. Trains a RandomForestClassifier
  5. Evaluates with accuracy, precision, recall, F1
  6. Extracts feature importance
  7. Saves the trained model + metrics + feature importance to ml/artifacts/

Run:
    python ml/train.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from sklearn.preprocessing import OneHotEncoder

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)


def load_and_join():
    suppliers = pd.read_csv(DATA_DIR / "suppliers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    inventory = pd.read_csv(DATA_DIR / "inventory.csv")

    # Only orders with a known outcome (exclude "In Transit")
    df = orders[orders["status"] != "In Transit"].copy()
    df["is_delayed"] = (df["status"] == "Delayed").astype(int)

    df = df.merge(
        suppliers[["supplier_id", "on_time_rate", "average_delivery_days",
                   "quality_score", "risk_score"]],
        on="supplier_id", how="left", suffixes=("", "_supplier")
    )
    df = df.merge(
        inventory[["product_id", "category"]],
        on="product_id", how="left"
    )

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["order_month"] = df["order_date"].dt.month

    # IMPORTANT (leakage prevention): on_time_rate / average_delivery_days here
    # are supplier-level HISTORICAL aggregates computed across many orders, not
    # derived from this specific row's own outcome, so they are legitimate
    # predictive features rather than leakage.
    return df


def build_features(df):
    numeric_cols = ["on_time_rate", "average_delivery_days", "quality_score",
                     "risk_score", "quantity", "order_month"]
    categorical_cols = ["category", "priority"]

    X_numeric = df[numeric_cols].fillna(df[numeric_cols].median())

    encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = encoder.fit_transform(df[categorical_cols])
    cat_feature_names = encoder.get_feature_names_out(categorical_cols)

    X = np.hstack([X_numeric.values, X_cat])
    feature_names = numeric_cols + list(cat_feature_names)
    y = df["is_delayed"].values

    return X, y, feature_names, encoder


def main():
    print("Loading and joining data...")
    df = load_and_join()
    print(f"Total labeled orders: {len(df)}  (delayed rate: {df['is_delayed'].mean():.2%})")

    X, y, feature_names, encoder = build_features(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training RandomForestClassifier...")
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "positive_rate_test": round(float(np.mean(y_test)), 4),
    }
    cm = confusion_matrix(y_test, y_pred).tolist()
    report = classification_report(y_test, y_pred, output_dict=True)

    importances = model.feature_importances_
    feature_importance = sorted(
        [{"feature": f, "importance": round(float(imp), 4)}
         for f, imp in zip(feature_names, importances)],
        key=lambda x: x["importance"], reverse=True
    )

    print("\n=== Evaluation Metrics (on held-out test set) ===")
    for k, v in metrics.items():
        print(f"  {k}: {v}")

    print("\n=== Top Feature Importances ===")
    for f in feature_importance[:8]:
        print(f"  {f['feature']}: {f['importance']}")

    # Save everything
    with open(ARTIFACT_DIR / "delay_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open(ARTIFACT_DIR / "encoder.pkl", "wb") as f:
        pickle.dump(encoder, f)
    with open(ARTIFACT_DIR / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)
    with open(ARTIFACT_DIR / "metrics.json", "w") as f:
        json.dump({
            "metrics": metrics,
            "confusion_matrix": cm,
            "classification_report": report,
            "feature_importance": feature_importance,
        }, f, indent=2)

    print(f"\nSaved model + metrics to {ARTIFACT_DIR}/")


if __name__ == "__main__":
    main()
