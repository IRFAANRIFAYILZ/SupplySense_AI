"""
anomaly_detection.py
--------------------
Detects unusual, sudden changes in supplier delivery performance.

Approach: rolling-window Z-score on each supplier's monthly average delivery
delay, PLUS an Isolation Forest pass over supplier-month aggregates as a
cross-check. Both are simple, explainable, and standard techniques (per the
project brief's preference for Z-score/IQR/Isolation Forest over anything
exotic).

Definition of an anomaly (delivery-delay anomaly):
  For a given supplier and month, compute the average delivery delay for
  orders in that month. Compare it to that supplier's OWN historical mean and
  std-dev (excluding the month in question). If the z-score exceeds a
  threshold (default 2.0), flag it as an anomaly.

Also flags:
  - Unusual order quantity (z-score on quantity within a product's history)
  - Sudden inventory drop is checked separately in analytics (inventory
    snapshots aren't time series in this synthetic dataset, so we flag
    products currently below reorder level as a "current-state" anomaly).

Run:
    python ml/anomaly_detection.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
ARTIFACT_DIR = Path(__file__).parent / "artifacts"
ARTIFACT_DIR.mkdir(exist_ok=True)

Z_THRESHOLD = 2.0


def supplier_monthly_delay_anomalies(orders_df, suppliers_df):
    df = orders_df[orders_df["status"] != "In Transit"].copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["delivery_delay"] = pd.to_numeric(df["delivery_delay"], errors="coerce").fillna(0)
    df["year_month"] = df["order_date"].dt.to_period("M")

    monthly = df.groupby(["supplier_id", "year_month"]).agg(
        avg_delay=("delivery_delay", "mean"),
        order_count=("order_id", "count"),
    ).reset_index()

    MIN_ORDERS_IN_MONTH = 3     # ignore months with too few orders to be meaningful
    MIN_MEANINGFUL_DELAY = 2.0  # ignore "anomalies" that are trivially small in absolute terms
    SIGMA_FLOOR = 0.5           # avoid exploding z-scores from near-zero historical variance

    anomalies = []
    for sid, group in monthly.groupby("supplier_id"):
        group = group.sort_values("year_month").reset_index(drop=True)
        if len(group) < 4:
            continue  # not enough history to judge "normal"
        for idx in range(len(group)):
            if group.loc[idx, "order_count"] < MIN_ORDERS_IN_MONTH:
                continue
            history = group.drop(index=idx)
            history = history[history["order_count"] >= MIN_ORDERS_IN_MONTH]
            if len(history) < 3:
                continue
            mu, sigma = history["avg_delay"].mean(), history["avg_delay"].std()
            sigma = max(sigma, SIGMA_FLOOR) if not np.isnan(sigma) else SIGMA_FLOOR
            current = group.loc[idx, "avg_delay"]
            if current < MIN_MEANINGFUL_DELAY:
                continue
            z = (current - mu) / sigma
            if z >= Z_THRESHOLD and current > mu:  # only flag sudden WORSENING
                supplier_name = suppliers_df.loc[
                    suppliers_df["supplier_id"] == sid, "supplier_name"
                ]
                anomalies.append({
                    "supplier_id": sid,
                    "supplier_name": supplier_name.values[0] if len(supplier_name) else sid,
                    "month": str(group.loc[idx, "year_month"]),
                    "normal_avg_delay_days": round(float(mu), 1),
                    "anomalous_avg_delay_days": round(float(current), 1),
                    "z_score": round(float(z), 2),
                    "type": "Delivery Delay Spike",
                })
    return sorted(anomalies, key=lambda x: x["z_score"], reverse=True)


def order_quantity_anomalies(orders_df, iqr_multiplier=3.0):
    """IQR-based detection of unusually large/small order quantities per product."""
    df = orders_df.copy()
    anomalies = []
    for pid, group in df.groupby("product_id"):
        if len(group) < 10:
            continue
        q1, q3 = group["quantity"].quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        upper = q3 + iqr_multiplier * iqr
        outliers = group[group["quantity"] > upper]
        for _, row in outliers.iterrows():
            anomalies.append({
                "order_id": row["order_id"],
                "product_id": pid,
                "supplier_id": row["supplier_id"],
                "quantity": int(row["quantity"]),
                "upper_bound_iqr": round(float(upper), 1),
                "type": "Unusual Order Quantity",
            })
    return anomalies


def isolation_forest_cross_check(orders_df, suppliers_df):
    """Cross-check: Isolation Forest over supplier-month aggregates
    (avg delay, delay count, order count) to confirm Z-score findings using
    a second, independent method."""
    df = orders_df[orders_df["status"] != "In Transit"].copy()
    df["order_date"] = pd.to_datetime(df["order_date"])
    df["delivery_delay"] = pd.to_numeric(df["delivery_delay"], errors="coerce").fillna(0)
    df["is_delayed"] = (df["status"] == "Delayed").astype(int)
    df["year_month"] = df["order_date"].dt.to_period("M").astype(str)

    agg = df.groupby(["supplier_id", "year_month"]).agg(
        avg_delay=("delivery_delay", "mean"),
        delayed_count=("is_delayed", "sum"),
        order_count=("order_id", "count"),
    ).reset_index()

    if len(agg) < 10:
        return []

    features = agg[["avg_delay", "delayed_count", "order_count"]].values
    iso = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    preds = iso.fit_predict(features)
    scores = iso.decision_function(features)

    agg["iso_flag"] = preds == -1
    agg["iso_score"] = scores

    flagged = agg[agg["iso_flag"]].sort_values("iso_score")
    results = []
    for _, row in flagged.iterrows():
        supplier_name = suppliers_df.loc[
            suppliers_df["supplier_id"] == row["supplier_id"], "supplier_name"
        ]
        results.append({
            "supplier_id": row["supplier_id"],
            "supplier_name": supplier_name.values[0] if len(supplier_name) else row["supplier_id"],
            "month": row["year_month"],
            "avg_delay": round(float(row["avg_delay"]), 1),
            "isolation_forest_score": round(float(row["iso_score"]), 3),
            "type": "Isolation Forest Flag (delay pattern)",
        })
    return results


def main():
    suppliers_df = pd.read_csv(DATA_DIR / "suppliers.csv")
    orders_df = pd.read_csv(DATA_DIR / "orders.csv")
    inventory_df = pd.read_csv(DATA_DIR / "inventory.csv")

    print("Running Z-score delivery-delay anomaly detection...")
    delay_anomalies = supplier_monthly_delay_anomalies(orders_df, suppliers_df)
    print(f"  Found {len(delay_anomalies)} delivery-delay anomalies")

    print("Running IQR order-quantity anomaly detection...")
    qty_anomalies = order_quantity_anomalies(orders_df)
    print(f"  Found {len(qty_anomalies)} order-quantity anomalies")

    print("Running Isolation Forest cross-check...")
    iso_anomalies = isolation_forest_cross_check(orders_df, suppliers_df)
    print(f"  Found {len(iso_anomalies)} isolation-forest-flagged supplier-months")

    low_stock = inventory_df[inventory_df["current_stock"] < inventory_df["reorder_level"]]
    low_stock_list = low_stock[["product_id", "product_name", "current_stock",
                                 "reorder_level"]].to_dict(orient="records")
    print(f"  Found {len(low_stock_list)} low-stock products (current-state check)")

    output = {
        "delivery_delay_anomalies": delay_anomalies,
        "order_quantity_anomalies": qty_anomalies,
        "isolation_forest_anomalies": iso_anomalies,
        "low_stock_products": low_stock_list,
    }
    with open(ARTIFACT_DIR / "anomalies.json", "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved anomaly report to {ARTIFACT_DIR / 'anomalies.json'}")

    if delay_anomalies:
        top = delay_anomalies[0]
        print("\n=== Example Anomaly ===")
        print(f"Supplier: {top['supplier_name']} ({top['supplier_id']})")
        print(f"Month: {top['month']}")
        print(f"Normal avg delay: {top['normal_avg_delay_days']} days")
        print(f"Anomalous avg delay: {top['anomalous_avg_delay_days']} days")
        print(f"Z-score: {top['z_score']}")


if __name__ == "__main__":
    main()
