"""
app.py
------
SupplySense AI -- main Streamlit application.

Run:
    streamlit run app.py
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from utils.helpers import load_csv_safe, format_currency, format_percent, inventory_value

BASE_DIR = Path(__file__).parent

st.set_page_config(
    page_title="SupplySense AI",
    page_icon="\U0001F4E6",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# THEME / STYLE
# A deliberate slate + amber palette (industrial / logistics feel) instead of
# the generic purple-gradient SaaS look. Deep slate for structure, a single
# amber accent for attention (risk, alerts), muted greens/reds only for
# status, not decoration.
# ---------------------------------------------------------------------------
PRIMARY = "#1F2937"      # slate-800, structural
ACCENT = "#D97706"       # amber-600, single accent color
GOOD = "#0F766E"         # teal-700
BAD = "#B91C1C"          # red-700
WARN = "#B45309"         # amber-700
BG_CARD = "#F8FAFC"

st.markdown(f"""
<style>
    .stApp {{ background-color: #FFFFFF; }}
    h1, h2, h3 {{ color: {PRIMARY}; font-family: 'Georgia', serif; }}
    div[data-testid="stMetric"] {{
        background-color: {BG_CARD};
        border: 1px solid #E2E8F0;
        border-left: 4px solid {ACCENT};
        border-radius: 4px;
        padding: 14px 16px;
    }}
    div[data-testid="stMetricLabel"] {{ color: #475569; font-size: 0.85rem; }}
    section[data-testid="stSidebar"] {{ background-color: {PRIMARY}; }}
    section[data-testid="stSidebar"] * {{ color: #F1F5F9 !important; }}
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_all_data():
    suppliers = load_csv_safe("suppliers.csv")
    orders = load_csv_safe("orders.csv")
    inventory = load_csv_safe("inventory.csv")
    orders["order_date"] = pd.to_datetime(orders["order_date"])
    return suppliers, orders, inventory


def risk_color(level):
    return {"HIGH": BAD, "MEDIUM": WARN, "LOW": GOOD}.get(level, "#94A3B8")


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
st.sidebar.title("SupplySense AI")
st.sidebar.caption("Intelligent Supply Chain Analytics & Decision Support")

PAGE = st.sidebar.radio(
    "Navigate",
    ["Dashboard", "Suppliers", "Orders", "Inventory", "Risk & Anomalies",
     "AI Assistant", "Document Intelligence", "ML Predictions", "Model Evaluation"],
    label_visibility="collapsed",
)

try:
    suppliers, orders, inventory = load_all_data()
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()


# ---------------------------------------------------------------------------
# DASHBOARD
# ---------------------------------------------------------------------------
if PAGE == "Dashboard":
    st.title("Executive Dashboard")
    st.caption("Company-wide supply chain health at a glance")

    completed = orders[orders["status"] != "In Transit"].copy()

    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Total Suppliers", len(suppliers))
    col2.metric("Active Suppliers", int(suppliers["active"].sum()))
    col3.metric("Avg Delivery Time", f"{suppliers['average_delivery_days'].mean():.1f} days")
    col4.metric("On-Time Delivery", format_percent(completed.eval("status == 'On Time'").mean()))
    col5.metric("Delayed Orders", int((completed["status"] == "Delayed").sum()))

    col6, col7, col8, col9 = st.columns(4)
    low_stock = inventory[inventory["current_stock"] < inventory["reorder_level"]]
    col6.metric("Low-Stock Products", len(low_stock))
    col7.metric("High-Risk Suppliers", int((suppliers["risk_level"] == "HIGH").sum()))
    col8.metric("Inventory Value", format_currency(inventory_value(inventory)))
    col9.metric("Medium-Risk Suppliers", int((suppliers["risk_level"] == "MEDIUM").sum()))

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Supplier Risk Distribution")
        risk_counts = suppliers["risk_level"].value_counts().reindex(["LOW", "MEDIUM", "HIGH"]).fillna(0)
        fig = px.bar(
            x=risk_counts.index, y=risk_counts.values,
            color=risk_counts.index,
            color_discrete_map={"LOW": GOOD, "MEDIUM": WARN, "HIGH": BAD},
            labels={"x": "Risk Level", "y": "Number of Suppliers"},
        )
        fig.update_layout(showlegend=False, plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        st.subheader("Order Status Breakdown")
        status_counts = orders["status"].value_counts()
        fig = px.pie(values=status_counts.values, names=status_counts.index,
                     color=status_counts.index,
                     color_discrete_map={"On Time": GOOD, "Delayed": BAD, "In Transit": "#94A3B8"})
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Delivery Delays Over Time")
        monthly = completed.copy()
        monthly["month"] = monthly["order_date"].dt.to_period("M").astype(str)
        monthly_delay = monthly.groupby("month").apply(
            lambda g: (g["status"] == "Delayed").mean() * 100, include_groups=False
        ).reset_index(name="delay_rate_pct")
        fig = px.line(monthly_delay, x="month", y="delay_rate_pct", markers=True)
        fig.update_traces(line_color=ACCENT)
        fig.update_layout(plot_bgcolor="white", yaxis_title="Delay Rate (%)", xaxis_title="Month")
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        st.subheader("Inventory: Stock vs Reorder Level by Category")
        cat_summary = inventory.groupby("category").agg(
            current_stock=("current_stock", "sum"),
            reorder_level=("reorder_level", "sum"),
        ).reset_index()
        fig = go.Figure()
        fig.add_bar(name="Current Stock", x=cat_summary["category"], y=cat_summary["current_stock"], marker_color=GOOD)
        fig.add_bar(name="Reorder Level", x=cat_summary["category"], y=cat_summary["reorder_level"], marker_color=ACCENT)
        fig.update_layout(barmode="group", plot_bgcolor="white")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Recent Alerts")
    alerts = []
    for _, row in suppliers[suppliers["risk_level"] == "HIGH"].iterrows():
        alerts.append(f"\U0001F534 **{row['supplier_name']}** is HIGH risk (score {row['risk_score']})")
    for _, row in low_stock.head(5).iterrows():
        alerts.append(f"\U0001F7E1 **{row['product_name']}** is below reorder level "
                       f"({row['current_stock']} / {row['reorder_level']})")
    if alerts:
        for a in alerts[:10]:
            st.markdown(a)
    else:
        st.info("No active alerts.")


# ---------------------------------------------------------------------------
# SUPPLIERS
# ---------------------------------------------------------------------------
elif PAGE == "Suppliers":
    st.title("Supplier Intelligence")

    col1, col2 = st.columns([1, 2])
    with col1:
        category_filter = st.multiselect("Filter by category", suppliers["category"].unique())
        risk_filter = st.multiselect("Filter by risk level", ["LOW", "MEDIUM", "HIGH"])
        search = st.text_input("Search supplier name")

    filtered = suppliers.copy()
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]
    if risk_filter:
        filtered = filtered[filtered["risk_level"].isin(risk_filter)]
    if search:
        filtered = filtered[filtered["supplier_name"].str.contains(search, case=False)]

    st.dataframe(
        filtered[["supplier_id", "supplier_name", "category", "location", "on_time_rate",
                  "average_delivery_days", "quality_score", "risk_level", "risk_score"]],
        use_container_width=True, height=300,
    )

    st.divider()
    st.subheader("Supplier Profile")
    selected_id = st.selectbox("Select a supplier", filtered["supplier_id"] if len(filtered) else suppliers["supplier_id"])
    if selected_id:
        row = suppliers[suppliers["supplier_id"] == selected_id].iloc[0]
        c1, c2, c3 = st.columns([2, 1, 1])
        with c1:
            st.markdown(f"### {row['supplier_name']}")
            st.caption(f"{row['category']} · {row['location']}")
            st.markdown(
                f"**On-time delivery:** {row['on_time_rate']*100:.1f}%  \n"
                f"**Average delay-adjusted delivery:** {row['average_delivery_days']} days  \n"
                f"**Quality score:** {row['quality_score']}  \n"
                f"**Orders:** {row['total_orders']}  \n"
                f"**Delayed orders:** {row['delayed_orders']}  \n"
                f"**Cost index:** {row['cost_index']}"
            )
        with c2:
            st.metric("Risk Level", row["risk_level"])
            st.metric("Risk Score", row["risk_score"])

        with st.expander("Why this risk level? (explanation)"):
            reasons = []
            if row["on_time_rate"] < 0.85:
                reasons.append(f"On-time delivery rate is {row['on_time_rate']*100:.1f}%, below the 85% target.")
            if row["average_delivery_days"] > 7:
                reasons.append(f"Average delivery time of {row['average_delivery_days']} days exceeds the 7-day baseline.")
            if row["quality_score"] < 85:
                reasons.append(f"Quality score of {row['quality_score']} is below the 85 target.")
            if row["cost_index"] > 115:
                reasons.append(f"Cost index of {row['cost_index']} is above the category norm (110).")
            if not reasons:
                reasons.append("No significant risk factors -- performance is within target on all dimensions.")
            for r in reasons:
                st.markdown(f"- {r}")

        if st.button("Generate AI Insight for this supplier"):
            supplier_orders = orders[(orders["supplier_id"] == selected_id) & (orders["status"] != "In Transit")].copy()
            supplier_orders = supplier_orders.sort_values("order_date")
            if len(supplier_orders) >= 10:
                half = len(supplier_orders) // 2
                first_rate = (supplier_orders.iloc[:half]["status"] == "On Time").mean()
                second_rate = (supplier_orders.iloc[half:]["status"] == "On Time").mean()
                change = (second_rate - first_rate) * 100
                direction = "increased" if change > 0 else "decreased"
                st.success(
                    f"**Supplier Analysis**\n\n"
                    f"The supplier's on-time delivery rate {direction} from "
                    f"{first_rate*100:.0f}% to {second_rate*100:.0f}% comparing the first and "
                    f"second half of its order history.\n\n"
                    + ("The primary concern is increasing delivery delay; the supplier should be "
                       "reviewed for potential capacity or logistics issues."
                       if change < -5 else
                       "Performance has been stable or improving; continue standard monitoring.")
                )
            else:
                st.info("Not enough order history for this supplier to compute a trend.")


# ---------------------------------------------------------------------------
# ORDERS
# ---------------------------------------------------------------------------
elif PAGE == "Orders":
    st.title("Orders")
    c1, c2, c3 = st.columns(3)
    status_filter = c1.multiselect("Status", orders["status"].unique())
    supplier_filter = c2.multiselect("Supplier", suppliers["supplier_name"].unique())
    priority_filter = c3.multiselect("Priority", orders["priority"].unique())

    filtered = orders.merge(suppliers[["supplier_id", "supplier_name"]], on="supplier_id")
    if status_filter:
        filtered = filtered[filtered["status"].isin(status_filter)]
    if supplier_filter:
        filtered = filtered[filtered["supplier_name"].isin(supplier_filter)]
    if priority_filter:
        filtered = filtered[filtered["priority"].isin(priority_filter)]

    st.metric("Matching Orders", len(filtered))
    st.dataframe(
        filtered[["order_id", "supplier_name", "product_id", "order_date", "quantity",
                  "priority", "expected_delivery", "actual_delivery", "delivery_delay", "status"]],
        use_container_width=True, height=450,
    )


# ---------------------------------------------------------------------------
# INVENTORY
# ---------------------------------------------------------------------------
elif PAGE == "Inventory":
    st.title("Inventory")
    c1, c2 = st.columns(2)
    c1.metric("Total Inventory Value", format_currency(inventory_value(inventory)))
    low_stock = inventory[inventory["current_stock"] < inventory["reorder_level"]]
    c2.metric("Low-Stock Products", len(low_stock))

    category_filter = st.multiselect("Filter by category", inventory["category"].unique())
    filtered = inventory.copy()
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]

    filtered = filtered.copy()
    filtered["status"] = np.where(filtered["current_stock"] < filtered["reorder_level"], "LOW STOCK", "OK")
    st.dataframe(filtered, use_container_width=True, height=450)

    st.subheader("Inventory Value by Category")
    filtered["value"] = filtered["current_stock"] * filtered["unit_cost"]
    by_cat = filtered.groupby("category")["value"].sum().reset_index()
    fig = px.bar(by_cat, x="category", y="value", color_discrete_sequence=[ACCENT])
    fig.update_layout(plot_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# RISK & ANOMALIES
# ---------------------------------------------------------------------------
elif PAGE == "Risk & Anomalies":
    st.title("Risk & Anomalies")
    st.caption("Z-score + Isolation Forest anomaly detection over supplier delivery performance")

    anomaly_path = BASE_DIR / "ml" / "artifacts" / "anomalies.json"
    if not anomaly_path.exists():
        st.warning("Anomaly report not found. Run `python ml/anomaly_detection.py` first.")
    else:
        report = json.loads(anomaly_path.read_text())

        tab1, tab2, tab3 = st.tabs(["Delivery Delay Spikes", "Isolation Forest Cross-Check", "Low Stock"])
        with tab1:
            st.markdown("Z-score method: flags supplier-months where average delay is "
                        "abnormally high **relative to that supplier's own history**.")
            for a in report["delivery_delay_anomalies"][:15]:
                st.markdown(
                    f"**\u26A0 {a['supplier_name']}** ({a['supplier_id']}) — {a['month']}  \n"
                    f"Normal avg delay: {a['normal_avg_delay_days']} days &nbsp;|&nbsp; "
                    f"Current: **{a['anomalous_avg_delay_days']} days** &nbsp;|&nbsp; "
                    f"Z-score: {a['z_score']}"
                )
                st.divider()
        with tab2:
            st.markdown("Independent cross-check using Isolation Forest over "
                        "(avg delay, delayed count, order count) per supplier-month.")
            iso_df = pd.DataFrame(report["isolation_forest_anomalies"])
            if len(iso_df):
                st.dataframe(iso_df, use_container_width=True)
            else:
                st.info("No isolation-forest anomalies found.")
        with tab3:
            low_df = pd.DataFrame(report["low_stock_products"])
            st.dataframe(low_df, use_container_width=True)


# ---------------------------------------------------------------------------
# AI ASSISTANT (RAG)
# ---------------------------------------------------------------------------
elif PAGE == "AI Assistant":
    st.title("AI Assistant")
    st.caption("Ask questions about company policy (RAG) or the live dataset (NL analytics)")

    tab1, tab2 = st.tabs(["Policy Q&A (RAG)", "Data Analytics (Natural Language)"])

    with tab1:
        st.markdown("Answers are generated **only** from the documents in `data/documents/` "
                    "and always show their sources.")
        q = st.text_input("Ask a policy question",
                          placeholder="What is the penalty for late delivery?")
        if q:
            from rag.qa import answer_question
            with st.spinner("Retrieving relevant policy sections..."):
                result = answer_question(q)
            st.markdown(f"**Answer** _(mode: {result['mode']})_")
            st.write(result["answer"])
            if result["sources"]:
                st.markdown("**Sources**")
                for s in result["sources"]:
                    st.markdown(f"- {s}")

    with tab2:
        st.markdown("Ask about the live supplier/order/inventory data. Answers come from "
                    "**controlled, pre-written analysis functions** — no arbitrary code is "
                    "ever executed.")
        q2 = st.text_input("Ask a data question",
                           placeholder="Which supplier had the highest number of delays?")
        if q2:
            from analytics.analysis import answer_data_question
            result = answer_data_question(q2)
            st.write(result["text"])
            if result.get("table") is not None:
                st.dataframe(result["table"], use_container_width=True)


# ---------------------------------------------------------------------------
# DOCUMENT INTELLIGENCE
# ---------------------------------------------------------------------------
elif PAGE == "Document Intelligence":
    st.title("Document Intelligence")
    st.caption("The fictional enterprise documents powering the RAG system")

    doc_dir = BASE_DIR / "data" / "documents"
    for path in sorted(doc_dir.glob("*.md")):
        with st.expander(path.stem.replace("_", " ").title()):
            st.markdown(path.read_text())


# ---------------------------------------------------------------------------
# ML PREDICTIONS
# ---------------------------------------------------------------------------
elif PAGE == "ML Predictions":
    st.title("ML Predictions — Delivery Delay Risk")
    st.caption("RandomForestClassifier trained on historical order outcomes")

    metrics_path = BASE_DIR / "ml" / "artifacts" / "metrics.json"
    if not metrics_path.exists():
        st.warning("Model not found. Run `python ml/train.py` first.")
    else:
        metrics = json.loads(metrics_path.read_text())["metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Accuracy", metrics["accuracy"])
        c2.metric("Precision", metrics["precision"])
        c3.metric("Recall", metrics["recall"])
        c4.metric("F1 Score", metrics["f1_score"])

        st.subheader("Feature Importance")
        fi = pd.DataFrame(json.loads(metrics_path.read_text())["feature_importance"])
        fig = px.bar(fi.head(10), x="importance", y="feature", orientation="h",
                     color_discrete_sequence=[ACCENT])
        fig.update_layout(plot_bgcolor="white", yaxis={"categoryorder": "total ascending"})
        st.plotly_chart(fig, use_container_width=True)

        st.divider()
        st.subheader("Try a Prediction")
        from ml.predict import DelayPredictor
        predictor = DelayPredictor()

        c1, c2, c3 = st.columns(3)
        sup_choice = c1.selectbox("Supplier", suppliers["supplier_name"])
        sup_row = suppliers[suppliers["supplier_name"] == sup_choice].iloc[0]
        quantity = c2.number_input("Order quantity", min_value=1, value=500)
        month = c3.selectbox("Order month", list(range(1, 13)), index=5)

        c4, c5 = st.columns(2)
        category = c4.selectbox("Category", inventory["category"].unique())
        priority = c5.selectbox("Priority", ["Standard", "High", "Urgent"])

        if st.button("Predict"):
            result = predictor.predict_for_supplier_row(sup_row, quantity, month, category, priority)
            if result["prediction"] == "Delayed":
                st.error(f"Prediction: **{result['prediction']}** "
                        f"(probability: {result['probability_delayed']*100:.1f}%)")
            else:
                st.success(f"Prediction: **{result['prediction']}** "
                          f"(probability of delay: {result['probability_delayed']*100:.1f}%)")


# ---------------------------------------------------------------------------
# MODEL EVALUATION
# ---------------------------------------------------------------------------
elif PAGE == "Model Evaluation":
    st.title("Model Evaluation")
    st.caption("Base LLM vs RAG vs QLoRA vs RAG+QLoRA — genuinely computed, not fabricated")

    eval_path = BASE_DIR / "llm" / "artifacts" / "evaluation_results.json"
    if not eval_path.exists():
        st.warning("Evaluation report not found. Run `python -m llm.evaluation` first.")
    else:
        results = json.loads(eval_path.read_text())
        st.metric(f"Retrieval Recall@{results['top_k']}", results["retrieval_recall_at_k"])

        st.subheader("Configuration Comparison")
        rag_b = results["configs"]["B_rag"]
        table_data = pd.DataFrame([
            {"Approach": "A. Base LLM", "Groundedness": "N/A", "Answer Relevance": "N/A",
             "Note": "Requires an LLM API key or local model — not run in this environment"},
            {"Approach": "B. RAG", "Groundedness": rag_b.get("groundedness"),
             "Answer Relevance": rag_b.get("answer_relevance"),
             "Note": f"backend: {rag_b.get('generation_backend')}"},
            {"Approach": "C. QLoRA", "Groundedness": "N/A", "Answer Relevance": "N/A",
             "Note": "Requires a trained adapter + GPU inference — see README"},
            {"Approach": "D. RAG + QLoRA", "Groundedness": "N/A", "Answer Relevance": "N/A",
             "Note": "Requires a trained adapter + GPU inference — see README"},
        ])
        st.dataframe(table_data, use_container_width=True, hide_index=True)

        with st.expander("Per-question retrieval detail"):
            st.dataframe(pd.DataFrame(results["per_question_retrieval"]), use_container_width=True)
