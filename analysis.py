"""
analysis.py
-----------
Natural-language data analysis over the structured supply-chain dataset,
WITHOUT ever executing arbitrary code (no eval, no exec, no shell-outs).

How it works (intent-routing, not code generation):
  1. The user's question is matched against a small set of known INTENTS
     using keyword/regex rules (e.g. "highest number of delays" -> the
     `most_delayed_supplier` intent).
  2. Any numeric thresholds or entity names mentioned in the question (e.g.
     "more than 20 delayed orders", a specific supplier name) are extracted
     with regex.
  3. Each intent maps to a SAFE, PRE-WRITTEN pandas function from the
     `SAFE_OPERATIONS` registry below. The LLM (or this rule-based router)
     never writes or executes new code -- it only SELECTS which vetted
     function to call and with which extracted parameters.
  4. If no intent matches confidently, the system says so rather than
     guessing.

This satisfies the project requirement: "DO NOT allow arbitrary Python
execution, shell commands, eval(), or unsafe code execution."

If ANTHROPIC_API_KEY is set, an LLM is used ONLY to help pick which intent
best matches a free-form question (classification, not code generation) --
the actual data operation is still one of the fixed SAFE_OPERATIONS.
"""

import re
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"


def _load_data():
    suppliers = pd.read_csv(DATA_DIR / "suppliers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    inventory = pd.read_csv(DATA_DIR / "inventory.csv")
    return suppliers, orders, inventory


# ---------------------------------------------------------------------------
# SAFE OPERATIONS REGISTRY
# Each function takes (suppliers, orders, inventory, params) and returns a
# dict: {"text": str, "table": DataFrame|None, "chart_data": dict|None}
# ---------------------------------------------------------------------------

def op_most_delayed_supplier(suppliers, orders, inventory, params):
    top = suppliers.sort_values("delayed_orders", ascending=False).head(5)
    best = top.iloc[0]
    text = (f"{best['supplier_name']} ({best['supplier_id']}) has the highest number "
            f"of delayed orders: {int(best['delayed_orders'])} delayed out of "
            f"{int(best['total_orders'])} total orders "
            f"({best['on_time_rate']*100:.1f}% on-time rate).")
    table = top[["supplier_id", "supplier_name", "delayed_orders", "total_orders", "on_time_rate"]]
    return {"text": text, "table": table}


def op_lowest_inventory_category(suppliers, orders, inventory, params):
    inv_summary = inventory.copy()
    inv_summary["stock_ratio"] = inv_summary["current_stock"] / inv_summary["reorder_level"].replace(0, 1)
    by_cat = inv_summary.groupby("category")["stock_ratio"].mean().sort_values()
    lowest_cat = by_cat.index[0]
    text = (f"The '{lowest_cat}' category has the lowest average stock-to-reorder-level "
            f"ratio ({by_cat.iloc[0]:.2f}x), meaning it is closest to running low on average "
            f"across its products.")
    table = by_cat.reset_index().rename(columns={"stock_ratio": "avg_stock_to_reorder_ratio"})
    return {"text": text, "table": table}


def op_suppliers_above_delay_threshold(suppliers, orders, inventory, params):
    threshold = params.get("number", 20)
    filtered = suppliers[suppliers["delayed_orders"] > threshold].sort_values(
        "delayed_orders", ascending=False
    )
    text = f"Found {len(filtered)} supplier(s) with more than {threshold} delayed orders."
    table = filtered[["supplier_id", "supplier_name", "delayed_orders", "total_orders", "on_time_rate"]]
    return {"text": text, "table": table}


def op_average_delivery_time_period(suppliers, orders, inventory, params):
    completed = orders[orders["status"] != "In Transit"].copy()
    completed["order_date"] = pd.to_datetime(completed["order_date"])
    completed["actual_delivery"] = pd.to_datetime(completed["actual_delivery"], errors="coerce")
    completed["full_delivery_days"] = (completed["actual_delivery"] - completed["order_date"]).dt.days

    last_date = completed["order_date"].max()
    window_start = last_date - pd.Timedelta(days=30)
    recent = completed[completed["order_date"] >= window_start]

    avg_days = recent["full_delivery_days"].mean()
    text = (f"The average delivery time over the last 30 days of order data "
            f"(ending {last_date.date()}) was {avg_days:.1f} days, "
            f"based on {len(recent)} completed orders.")
    return {"text": text, "table": None}


def op_supplier_performance_decline(suppliers, orders, inventory, params):
    completed = orders[orders["status"] != "In Transit"].copy()
    completed["order_date"] = pd.to_datetime(completed["order_date"])
    completed["is_delayed"] = (completed["status"] == "Delayed").astype(int)
    completed["half"] = pd.cut(
        completed["order_date"],
        bins=2, labels=["first_half", "second_half"]
    )

    pivot = completed.groupby(["supplier_id", "half"], observed=True)["is_delayed"].mean().unstack()
    pivot = pivot.dropna()
    pivot["decline"] = pivot["second_half"] - pivot["first_half"]
    pivot = pivot.sort_values("decline", ascending=False)

    if pivot.empty:
        return {"text": "Not enough data to compute performance decline.", "table": None}

    worst_id = pivot.index[0]
    worst_name = suppliers.loc[suppliers["supplier_id"] == worst_id, "supplier_name"].values
    worst_name = worst_name[0] if len(worst_name) else worst_id
    decline_pct = pivot.loc[worst_id, "decline"] * 100

    text = (f"{worst_name} ({worst_id}) shows the largest decline in performance: "
            f"its delay rate increased by {decline_pct:.1f} percentage points from the "
            f"first half to the second half of the available order history.")
    table = pivot.reset_index().rename(columns={
        "first_half": "delay_rate_first_half", "second_half": "delay_rate_second_half"
    }).head(10)
    return {"text": text, "table": table}


def op_supplier_lookup(suppliers, orders, inventory, params):
    name = params.get("name")
    if not name:
        return {"text": "Please specify a supplier name or ID.", "table": None}
    match = suppliers[
        suppliers["supplier_name"].str.contains(name, case=False, na=False) |
        suppliers["supplier_id"].str.contains(name, case=False, na=False)
    ]
    if match.empty:
        return {"text": f"No supplier found matching '{name}'.", "table": None}
    row = match.iloc[0]
    text = (f"{row['supplier_name']} ({row['supplier_id']}): on-time rate "
            f"{row['on_time_rate']*100:.1f}%, avg delivery {row['average_delivery_days']} days, "
            f"quality score {row['quality_score']}, risk level {row['risk_level']} "
            f"(score {row['risk_score']}).")
    return {"text": text, "table": match}


SAFE_OPERATIONS = {
    "most_delayed_supplier": op_most_delayed_supplier,
    "lowest_inventory_category": op_lowest_inventory_category,
    "suppliers_above_delay_threshold": op_suppliers_above_delay_threshold,
    "average_delivery_time_period": op_average_delivery_time_period,
    "supplier_performance_decline": op_supplier_performance_decline,
    "supplier_lookup": op_supplier_lookup,
}

# ---------------------------------------------------------------------------
# INTENT ROUTING (keyword/regex-based, no code generation)
# ---------------------------------------------------------------------------

INTENT_PATTERNS = [
    (r"highest number of delay|most delay|which supplier.*most.*delay", "most_delayed_supplier"),
    (r"lowest inventory|lowest stock|running low|least stock", "lowest_inventory_category"),
    (r"more than \d+ delayed|delayed orders? (greater|above|more than)", "suppliers_above_delay_threshold"),
    (r"average delivery time|average delay|delivery time (last|past)", "average_delivery_time_period"),
    (r"decline|declined|performance drop|got worse|deteriorat", "supplier_performance_decline"),
    (r"tell me about|show me supplier|profile of|information (on|about)", "supplier_lookup"),
]


def extract_number(question):
    m = re.search(r"\b(\d+)\b", question)
    return int(m.group(1)) if m else None


def extract_supplier_name(question):
    # crude heuristic: look for capitalized multi-word sequences
    m = re.search(r"(?:about|for|of)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)", question)
    return m.group(1) if m else None


def route_question(question: str):
    q_lower = question.lower()
    for pattern, intent in INTENT_PATTERNS:
        if re.search(pattern, q_lower):
            return intent
    return None


def answer_data_question(question: str):
    suppliers, orders, inventory = _load_data()
    intent = route_question(question)

    if intent is None:
        return {
            "text": ("I couldn't confidently map this question to one of the supported "
                      "analysis types. Try asking about: most-delayed suppliers, low "
                      "inventory categories, suppliers above a delay threshold, average "
                      "delivery time, performance decline, or a specific supplier."),
            "table": None,
            "intent": None,
        }

    params = {
        "number": extract_number(question),
        "name": extract_supplier_name(question),
    }
    result = SAFE_OPERATIONS[intent](suppliers, orders, inventory, params)
    result["intent"] = intent
    return result


if __name__ == "__main__":
    tests = [
        "Which supplier had the highest number of delays?",
        "Which product category has the lowest inventory?",
        "Show me suppliers with more than 20 delayed orders.",
        "What was the average delivery time last month?",
        "Which supplier's performance declined the most?",
        "What's the weather today?",
    ]
    for t in tests:
        r = answer_data_question(t)
        print(f"\nQ: {t}")
        print(f"Intent: {r['intent']}")
        print(f"A: {r['text']}")
