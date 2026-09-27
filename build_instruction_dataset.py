"""
build_instruction_dataset.py
-----------------------------
Builds the domain-specific instruction dataset used to fine-tune the LLM with
LoRA/QLoRA. Unlike a hand-written toy dataset, these examples are generated
FROM the actual suppliers.csv / orders.csv / inventory.csv data, so the
target responses are genuinely grounded in real (synthetic) numbers rather
than invented. Templates vary the phrasing so the model learns the desired
*style and structure* (concise, structured, business-appropriate), which is
exactly what LoRA is being used for here -- RAG already supplies facts, LoRA
supplies tone/format.

Produces four instruction types, matching the project brief:
  - Supply-chain analysis (trend description)
  - Supplier summaries
  - Risk explanations
  - Operational recommendations / structured business responses

Output: llm/data/instruction_dataset.jsonl
Each line: {"instruction": ..., "input": ..., "output": ...}
(this is the standard Alpaca-style format, easy to load with `datasets`)

Run:
    python llm/build_instruction_dataset.py
"""

import json
import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
OUT_DIR = Path(__file__).parent / "data"
OUT_DIR.mkdir(exist_ok=True)

random.seed(42)


def risk_reasons(row):
    reasons = []
    if row["on_time_rate"] < 0.85:
        reasons.append(f"an on-time delivery rate of only {row['on_time_rate']*100:.0f}%")
    if row["average_delivery_days"] > 7:
        reasons.append(f"a longer-than-average delivery time of {row['average_delivery_days']} days")
    if row["quality_score"] < 85:
        reasons.append(f"a below-target quality score of {row['quality_score']}")
    if row["cost_index"] > 115:
        reasons.append(f"an above-average cost index of {row['cost_index']}")
    if not reasons:
        reasons.append("consistently strong delivery and quality performance")
    return reasons


def recommendation_for_risk(level):
    if level == "HIGH":
        return ("1. Initiate a formal performance review.\n"
                "2. Contact the supplier regarding delivery and quality performance.\n"
                "3. Identify and qualify backup suppliers in the same category.\n"
                "4. Review contract performance clauses and penalty terms.")
    if level == "MEDIUM":
        return ("1. Monitor the supplier's performance over the next quarter.\n"
                "2. Request a corrective action plan if delays continue.\n"
                "3. Compare cost and quality against category peers.")
    return ("1. Continue standard quarterly evaluation.\n"
            "2. Consider increasing order volume given strong performance.\n"
            "3. No immediate corrective action required.")


def build_examples():
    suppliers = pd.read_csv(DATA_DIR / "suppliers.csv")
    orders = pd.read_csv(DATA_DIR / "orders.csv")
    inventory = pd.read_csv(DATA_DIR / "inventory.csv")

    examples = []

    # --- Supplier summaries ---
    summary_instructions = [
        "Summarize this supplier's performance in a few sentences.",
        "Give a brief performance summary for this supplier.",
        "Provide a concise supplier profile summary.",
    ]
    for _, row in suppliers.iterrows():
        instr = random.choice(summary_instructions)
        inp = (f"Supplier: {row['supplier_name']} ({row['supplier_id']})\n"
               f"On-time delivery: {row['on_time_rate']*100:.0f}%\n"
               f"Average delivery time: {row['average_delivery_days']} days\n"
               f"Quality score: {row['quality_score']}\n"
               f"Total orders: {row['total_orders']}, Delayed orders: {row['delayed_orders']}\n"
               f"Risk level: {row['risk_level']}")
        out = (f"{row['supplier_name']} has completed {row['total_orders']} orders with an "
               f"on-time delivery rate of {row['on_time_rate']*100:.0f}% and an average delivery "
               f"time of {row['average_delivery_days']} days. Quality score is {row['quality_score']}, "
               f"and {row['delayed_orders']} of its orders have been delayed. Overall risk is "
               f"classified as {row['risk_level']}.")
        examples.append({"instruction": instr, "input": inp, "output": out})

    # --- Risk explanations ---
    risk_instructions = [
        "Explain why this supplier received its risk classification.",
        "Why is this supplier rated at this risk level?",
        "Provide a risk explanation for this supplier.",
    ]
    for _, row in suppliers.iterrows():
        instr = random.choice(risk_instructions)
        inp = (f"Supplier: {row['supplier_name']}\n"
               f"Risk level: {row['risk_level']} (score: {row['risk_score']})\n"
               f"On-time rate: {row['on_time_rate']*100:.0f}%, "
               f"Avg delivery: {row['average_delivery_days']} days, "
               f"Quality: {row['quality_score']}, Cost index: {row['cost_index']}")
        reasons = risk_reasons(row)
        out = (f"{row['supplier_name']} is classified as {row['risk_level']} risk primarily due to "
               f"{', and '.join(reasons)}. " +
               ("This combination of factors indicates elevated operational risk."
                if row["risk_level"] != "LOW" else
                "These factors indicate a reliable, low-risk supplier relationship."))
        examples.append({"instruction": instr, "input": inp, "output": out})

    # --- Operational recommendations ---
    rec_instructions = [
        "Recommend next steps for managing this supplier.",
        "What actions should the procurement team take for this supplier?",
        "Provide operational recommendations based on this supplier's risk level.",
    ]
    for _, row in suppliers.iterrows():
        instr = random.choice(rec_instructions)
        inp = f"Supplier: {row['supplier_name']}\nRisk level: {row['risk_level']}"
        out = f"Supplier Risk: {row['risk_level']}\n\nPossible actions:\n{recommendation_for_risk(row['risk_level'])}"
        examples.append({"instruction": instr, "input": inp, "output": out})

    # --- Supply-chain trend / analysis statements ---
    completed = orders[orders["status"] != "In Transit"].copy()
    completed["order_date"] = pd.to_datetime(completed["order_date"])
    completed["is_delayed"] = (completed["status"] == "Delayed").astype(int)
    trend_instructions = [
        "Describe this supplier's delivery performance trend.",
        "Analyze the change in this supplier's delivery reliability over time.",
    ]
    for sid, group in completed.groupby("supplier_id"):
        group = group.sort_values("order_date")
        if len(group) < 20:
            continue
        half = len(group) // 2
        first_rate = 1 - group.iloc[:half]["is_delayed"].mean()
        second_rate = 1 - group.iloc[half:]["is_delayed"].mean()
        name = suppliers.loc[suppliers["supplier_id"] == sid, "supplier_name"].values[0]
        change = (second_rate - first_rate) * 100
        instr = random.choice(trend_instructions)
        inp = f"Supplier: {name}\nEarlier on-time rate: {first_rate*100:.0f}%\nRecent on-time rate: {second_rate*100:.0f}%"
        direction = "improved" if change > 1 else ("declined" if change < -1 else "remained stable")
        out = (f"{name}'s on-time delivery rate {direction} from {first_rate*100:.0f}% to "
               f"{second_rate*100:.0f}% ({change:+.1f} percentage points). " +
               ("This warrants a closer review of recent operational or logistics changes."
                if direction == "declined" else
                "This trend should continue to be monitored in the next evaluation cycle."))
        examples.append({"instruction": instr, "input": inp, "output": out})

    random.shuffle(examples)
    return examples


def main():
    examples = build_examples()
    out_path = OUT_DIR / "instruction_dataset.jsonl"
    with open(out_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    n = len(examples)
    train_n = int(n * 0.9)
    train, val = examples[:train_n], examples[train_n:]
    with open(OUT_DIR / "train.jsonl", "w") as f:
        for ex in train:
            f.write(json.dumps(ex) + "\n")
    with open(OUT_DIR / "val.jsonl", "w") as f:
        for ex in val:
            f.write(json.dumps(ex) + "\n")

    print(f"Built {n} instruction examples ({len(train)} train / {len(val)} val).")
    print(f"Saved to {out_path}, train.jsonl, val.jsonl")
    print("\nExample:")
    print(json.dumps(examples[0], indent=2))


if __name__ == "__main__":
    main()
