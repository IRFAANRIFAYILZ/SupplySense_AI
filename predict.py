"""
predict.py
----------
Loads the trained delay-prediction model (from ml/artifacts/) and exposes a
simple function to score new/hypothetical orders. Used by the Streamlit app's
"ML Predictions" page.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACT_DIR = Path(__file__).parent / "artifacts"


class DelayPredictor:
    def __init__(self):
        with open(ARTIFACT_DIR / "delay_model.pkl", "rb") as f:
            self.model = pickle.load(f)
        with open(ARTIFACT_DIR / "encoder.pkl", "rb") as f:
            self.encoder = pickle.load(f)
        with open(ARTIFACT_DIR / "feature_names.json") as f:
            self.feature_names = json.load(f)
        with open(ARTIFACT_DIR / "metrics.json") as f:
            self.metrics = json.load(f)

    def predict_one(self, on_time_rate, average_delivery_days, quality_score,
                     risk_score, quantity, order_month, category, priority):
        numeric = np.array([[on_time_rate, average_delivery_days, quality_score,
                              risk_score, quantity, order_month]])
        cat_df = pd.DataFrame([[category, priority]], columns=["category", "priority"])
        cat_encoded = self.encoder.transform(cat_df)
        X = np.hstack([numeric, cat_encoded])

        proba = self.model.predict_proba(X)[0]
        # proba index 1 = P(delayed)
        p_delay = float(proba[1]) if len(proba) > 1 else float(proba[0])
        prediction = "Delayed" if p_delay >= 0.5 else "On Time"
        return {"prediction": prediction, "probability_delayed": round(p_delay, 3)}

    def predict_for_supplier_row(self, supplier_row, quantity, order_month, category, priority):
        """Convenience wrapper taking a row (dict or Series) from suppliers.csv."""
        return self.predict_one(
            on_time_rate=supplier_row["on_time_rate"],
            average_delivery_days=supplier_row["average_delivery_days"],
            quality_score=supplier_row["quality_score"],
            risk_score=supplier_row["risk_score"],
            quantity=quantity,
            order_month=order_month,
            category=category,
            priority=priority,
        )

    def get_metrics(self):
        return self.metrics


if __name__ == "__main__":
    # Quick smoke test
    predictor = DelayPredictor()
    result = predictor.predict_one(
        on_time_rate=0.65, average_delivery_days=11.5, quality_score=78,
        risk_score=48, quantity=800, order_month=12,
        category="Electronics", priority="Urgent"
    )
    print("Sample prediction (risky supplier, peak season, urgent order):", result)

    result2 = predictor.predict_one(
        on_time_rate=0.95, average_delivery_days=4.0, quality_score=93,
        risk_score=8, quantity=100, order_month=4,
        category="Office Supplies", priority="Standard"
    )
    print("Sample prediction (reliable supplier, off-season, standard order):", result2)
