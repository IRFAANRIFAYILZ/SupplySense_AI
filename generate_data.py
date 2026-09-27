"""
generate_data.py
----------------
Generates realistic SYNTHETIC (fictional) supply-chain data for SupplySense AI:
  - suppliers.csv
  - orders.csv
  - inventory.csv

The data is randomly generated but seeded for reproducibility, and deliberately
includes noise, imperfect suppliers, seasonal effects, and a handful of
injected anomalies (sudden delay spikes) so that the ML and anomaly-detection
components downstream have real signal to find.

Run:
    python data/generate_data.py
"""

import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

RNG_SEED = 42
rng = np.random.default_rng(RNG_SEED)

OUT_DIR = Path(__file__).parent

CATEGORIES = ["Electronics", "Raw Materials", "Packaging", "Machinery Parts",
              "Textiles", "Chemicals", "Office Supplies"]

LOCATIONS = ["Chennai, IN", "Shenzhen, CN", "Ho Chi Minh City, VN", "Hamburg, DE",
             "Detroit, US", "Bangalore, IN", "Busan, KR", "Monterrey, MX",
             "Rotterdam, NL", "Jakarta, ID"]

N_SUPPLIERS = 60
N_PRODUCTS = 45
N_ORDERS = 4000

# Suppliers we will deliberately make "high risk" (chronic late deliveries)
N_CHRONIC_LATE = 8
# Suppliers we will give a SUDDEN anomaly (used to demonstrate anomaly detection)
ANOMALY_SUPPLIERS = ["SUP0007", "SUP0023", "SUP0041"]


def generate_suppliers():
    supplier_ids = [f"SUP{str(i).zfill(4)}" for i in range(1, N_SUPPLIERS + 1)]
    name_roots = ["Global", "Prime", "Apex", "Summit", "Nova", "Pioneer", "Vertex",
                  "Atlas", "Horizon", "Meridian", "Union", "Pacific", "Sterling",
                  "Crest", "Ironclad", "Vantage", "Beacon", "Cascade", "Orbit", "Delta"]
    name_suffix = ["Parts", "Industries", "Supply Co.", "Materials", "Components",
                   "Manufacturing", "Trading", "Logistics", "Group", "Ltd."]

    rows = []
    for i, sid in enumerate(supplier_ids):
        chronic_late = i < N_CHRONIC_LATE  # first N are structurally worse

        base_delivery = rng.normal(6, 1.5) if not chronic_late else rng.normal(11, 2.5)
        base_delivery = max(1.5, base_delivery)

        on_time_rate = rng.normal(0.90, 0.05) if not chronic_late else rng.normal(0.60, 0.08)
        on_time_rate = float(np.clip(on_time_rate, 0.30, 0.99))

        total_orders = int(rng.integers(20, 220))
        delayed_orders = int(round(total_orders * (1 - on_time_rate) * rng.uniform(0.85, 1.15)))
        delayed_orders = min(delayed_orders, total_orders)

        quality_score = rng.normal(90, 6) if not chronic_late else rng.normal(76, 8)
        quality_score = float(np.clip(quality_score, 40, 100))

        cost_index = float(np.clip(rng.normal(100, 15), 60, 160))

        rows.append({
            "supplier_id": sid,
            "supplier_name": f"{rng.choice(name_roots)} {rng.choice(name_suffix)}",
            "category": rng.choice(CATEGORIES),
            "location": rng.choice(LOCATIONS),
            "quality_score": round(quality_score, 1),
            "average_delivery_days": round(base_delivery, 1),
            "on_time_rate": round(on_time_rate, 3),
            "total_orders": total_orders,
            "delayed_orders": delayed_orders,
            "cost_index": round(cost_index, 1),
            "active": bool(rng.random() > 0.05),
        })

    df = pd.DataFrame(rows)

    # Simple, transparent, rule-based risk score (0-100, higher = riskier).
    # This is documented and explainable -- not a black box.
    def risk_score(r):
        score = 0.0
        score += (1 - r["on_time_rate"]) * 45          # late deliveries matter most
        score += max(0, (r["average_delivery_days"] - 5)) * 2.2  # slow baseline delivery
        score += max(0, (85 - r["quality_score"])) * 0.6         # quality shortfall
        score += max(0, (r["cost_index"] - 110)) * 0.15          # expensive suppliers, minor factor
        return round(min(100, score), 1)

    df["risk_score"] = df.apply(risk_score, axis=1)

    def risk_level(s):
        if s >= 55:
            return "HIGH"
        elif s >= 30:
            return "MEDIUM"
        return "LOW"

    df["risk_level"] = df["risk_score"].apply(risk_level)
    return df


def generate_inventory():
    rows = []
    for i in range(1, N_PRODUCTS + 1):
        pid = f"PRD{str(i).zfill(4)}"
        category = rng.choice(CATEGORIES)
        monthly_demand = int(rng.integers(50, 3000))
        reorder_level = int(monthly_demand * rng.uniform(0.15, 0.35))
        # Some products deliberately under-stocked to feed dashboard "low stock" alerts
        under_stocked = rng.random() < 0.18
        if under_stocked:
            current_stock = int(reorder_level * rng.uniform(0.2, 0.9))
        else:
            current_stock = int(reorder_level * rng.uniform(1.0, 4.0))
        unit_cost = round(float(rng.uniform(2, 500)), 2)

        rows.append({
            "product_id": pid,
            "product_name": f"{category[:4].upper()}-{i:03d}",
            "category": category,
            "current_stock": current_stock,
            "reorder_level": reorder_level,
            "monthly_demand": monthly_demand,
            "unit_cost": unit_cost,
        })
    return pd.DataFrame(rows)


def generate_orders(suppliers_df, inventory_df):
    start_date = datetime(2024, 10, 1)
    end_date = datetime(2026, 9, 27)  # "today" in this fictional timeline
    date_range_days = (end_date - start_date).days

    supplier_ids = suppliers_df["supplier_id"].tolist()
    supplier_lookup = suppliers_df.set_index("supplier_id")
    product_ids = inventory_df["product_id"].tolist()

    rows = []
    for i in range(1, N_ORDERS + 1):
        oid = f"ORD{str(i).zfill(6)}"
        sid = rng.choice(supplier_ids)
        srow = supplier_lookup.loc[sid]

        order_day_offset = int(rng.integers(0, date_range_days))
        order_date = start_date + timedelta(days=order_day_offset)
        month = order_date.month

        # Seasonal effect: Nov-Jan (holiday season) has more delay pressure
        seasonal_factor = 1.35 if month in (11, 12, 1) else 1.0

        quantity = int(rng.integers(10, 2000))
        pid = rng.choice(product_ids)

        base_lead_time = srow["average_delivery_days"]
        expected_delivery = order_date + timedelta(days=round(base_lead_time))

        # Determine if this order is delayed, using supplier's on-time rate
        # plus seasonal effect plus a chance of an injected anomaly window.
        p_delay = (1 - srow["on_time_rate"]) * seasonal_factor
        p_delay = min(0.95, p_delay)

        # Injected anomaly: for ANOMALY_SUPPLIERS, orders placed in a specific
        # 30-day window suddenly have much higher delay probability & magnitude,
        # simulating a real "sudden performance degradation" event.
        anomaly_window = (order_date >= datetime(2026, 6, 1)) and (order_date <= datetime(2026, 6, 30))
        forced_anomaly = sid in ANOMALY_SUPPLIERS and anomaly_window
        if forced_anomaly:
            p_delay = 0.9

        is_delayed = rng.random() < p_delay

        if is_delayed:
            if forced_anomaly:
                delay_days = int(rng.integers(6, 14))
            else:
                delay_days = int(rng.integers(1, 8))
            actual_delivery = expected_delivery + timedelta(days=delay_days)
            status = "Delayed"
        else:
            # occasionally early delivery
            early = rng.integers(-1, 1)
            actual_delivery = expected_delivery + timedelta(days=int(early))
            delay_days = max(0, (actual_delivery - expected_delivery).days)
            status = "On Time"

        # Some orders still in transit / not yet delivered (near "today")
        if (end_date - order_date).days < int(base_lead_time) + 2:
            status = "In Transit"
            actual_delivery = pd.NaT
            delay_days = np.nan

        priority = rng.choice(["Standard", "High", "Urgent"], p=[0.7, 0.22, 0.08])

        rows.append({
            "order_id": oid,
            "supplier_id": sid,
            "product_id": pid,
            "order_date": order_date.date().isoformat(),
            "quantity": quantity,
            "priority": priority,
            "expected_delivery": expected_delivery.date().isoformat(),
            "actual_delivery": actual_delivery.date().isoformat() if pd.notna(actual_delivery) else "",
            "delivery_delay": delay_days if pd.notna(delay_days) else "",
            "status": status,
        })

    return pd.DataFrame(rows)


def main():
    print("Generating suppliers...")
    suppliers_df = generate_suppliers()

    print("Generating inventory...")
    inventory_df = generate_inventory()

    print("Generating orders (this uses supplier + inventory data)...")
    orders_df = generate_orders(suppliers_df, inventory_df)

    # Recompute total_orders / delayed_orders on suppliers from the actual
    # generated order data, so numbers are internally CONSISTENT (not just
    # independently-sampled fake fields).
    completed = orders_df[orders_df["status"] != "In Transit"].copy()
    completed["is_delayed"] = completed["status"] == "Delayed"
    agg = completed.groupby("supplier_id").agg(
        total_orders=("order_id", "count"),
        delayed_orders=("is_delayed", "sum"),
    ).reset_index()

    suppliers_df = suppliers_df.drop(columns=["total_orders", "delayed_orders"]).merge(
        agg, on="supplier_id", how="left"
    )
    suppliers_df["total_orders"] = suppliers_df["total_orders"].fillna(0).astype(int)
    suppliers_df["delayed_orders"] = suppliers_df["delayed_orders"].fillna(0).astype(int)
    suppliers_df["on_time_rate"] = suppliers_df.apply(
        lambda r: round(1 - (r["delayed_orders"] / r["total_orders"]), 3) if r["total_orders"] > 0 else 0.0,
        axis=1,
    )

    # Full delivery time = actual_delivery - order_date (in days), for completed orders.
    completed["order_date_dt"] = pd.to_datetime(completed["order_date"])
    completed["actual_delivery_dt"] = pd.to_datetime(completed["actual_delivery"])
    completed["full_delivery_days"] = (completed["actual_delivery_dt"] - completed["order_date_dt"]).dt.days

    avg_full_delivery = completed.groupby("supplier_id")["full_delivery_days"].mean()
    suppliers_df["average_delivery_days"] = suppliers_df["supplier_id"].map(avg_full_delivery).fillna(
        suppliers_df["average_delivery_days"]
    ).round(1)

    # Recompute risk score & level after consistency fix
    def risk_score(r):
        score = 0.0
        score += (1 - r["on_time_rate"]) * 45
        score += max(0, (r["average_delivery_days"] - 5)) * 2.2
        score += max(0, (85 - r["quality_score"])) * 0.6
        score += max(0, (r["cost_index"] - 110)) * 0.15
        return round(min(100, score), 1)

    suppliers_df["risk_score"] = suppliers_df.apply(risk_score, axis=1)
    suppliers_df["risk_level"] = suppliers_df["risk_score"].apply(
        lambda s: "HIGH" if s >= 55 else ("MEDIUM" if s >= 30 else "LOW")
    )

    suppliers_df.to_csv(OUT_DIR / "suppliers.csv", index=False)
    inventory_df.to_csv(OUT_DIR / "inventory.csv", index=False)
    orders_df.to_csv(OUT_DIR / "orders.csv", index=False)

    print(f"suppliers.csv: {len(suppliers_df)} rows")
    print(f"inventory.csv: {len(inventory_df)} rows")
    print(f"orders.csv:    {len(orders_df)} rows")
    print("Done.")


if __name__ == "__main__":
    main()
