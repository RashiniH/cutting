"""
🔐 ConfirmIQ – Commitment Risk Engine
=====================================
Predicts which bookings, appointments, orders and repayments will break —
before they cost Sri Lankan businesses money.

Run with:
    streamlit run confirmiq_dashboard.py

WHAT CHANGED IN THIS VERSION
----------------------------
1. REAL risk engine: Risk Score is now COMPUTED from each commitment's
   features (lead time, prior no-shows, payment method, etc.) instead of
   being a random number. Scores are reproducible and explainable.
2. Real explanations: the "Why this score?" panel shows the actual factors
   that drove THIS commitment's score, with their point contributions.
3. Cached, seeded data: data no longer flickers/changes on every click.
   A sidebar control lets you regenerate the sample.
4. New "Score a Commitment" page: enter a commitment's details and get a
   live score, gauge, recommended action and factor breakdown.
5. CSV export on the industry and analytics pages.
6. Status is now consistent with the computed risk (high score -> At Risk).
"""

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# ---------------------------------------------------------------------------
# Palette (kept identical to the original design)
# ---------------------------------------------------------------------------
PRIMARY = "#00B8A9"
SECONDARY = "#003D5B"
RED = "#EE5A6F"
ORANGE = "#F7A072"
GREEN = "#26C485"

INDUSTRIES = ["Hospitality", "Healthcare", "E-commerce", "Banking", "Dining & Delivery"]


# ===========================================================================
# 1. RISK ENGINE  (pure functions – no Streamlit, fully testable)
# ===========================================================================
def _clip(x):
    """Keep scores in a sane, presentable 2–98 band and return an int."""
    return int(max(2, min(98, round(x))))


def score_hospitality(f):
    """f: dict with lead_days, duration, amount, room_type, prev_cancel."""
    factors = []
    base = 12
    lt = f["lead_days"] * 1.6
    factors.append(("Booking lead time", lt,
                    f"{f['lead_days']} days until check-in "
                    f"({'long lead time raises cancellation risk' if f['lead_days'] >= 14 else 'short lead time'})"))

    single = 12 if f["duration"] == 1 else (-5 if f["duration"] >= 7 else 0)
    factors.append(("Stay length", single,
                    f"{f['duration']}-night stay "
                    f"({'single nights are no-show prone' if f['duration'] == 1 else 'longer stays are sticky'})"))

    pc = f["prev_cancel"] * 12
    factors.append(("Cancellation history", pc,
                    f"{f['prev_cancel']} prior cancellation(s)"))

    room = {"Suite": 6, "Deluxe": 3, "Standard": 0}[f["room_type"]]
    factors.append(("Room type", room, f"{f['room_type']} (higher-value rooms carry more exposure)"))

    amt = (f["amount"] - 15000) / 85000 * 10
    factors.append(("Booking value", amt, f"Rs {f['amount']:,}"))

    return _clip(base + lt + single + pc + room + amt), factors


def score_healthcare(f):
    """f: days_until, prev_no_shows, specialist, time_slot."""
    factors = []
    base = 14
    du = f["days_until"] * 0.9
    factors.append(("Days until appointment", du,
                    f"{f['days_until']} days out "
                    f"({'distant appointments are forgotten' if f['days_until'] >= 21 else 'near-term'})"))

    ns = f["prev_no_shows"] * 16
    factors.append(("No-show history", ns,
                    f"{f['prev_no_shows']} previous no-show(s) "
                    f"({'strong predictor' if f['prev_no_shows'] else 'clean record'})"))

    spec = {"Dentistry": 8, "Neurology": 4, "Orthopedics": 4, "Cardiology": 2}[f["specialist"]]
    factors.append(("Specialty", spec, f"{f['specialist']}"))

    slot = {"9:00 AM": 6, "4:00 PM": 4, "11:00 AM": 0, "2:00 PM": 0}[f["time_slot"]]
    factors.append(("Time slot", slot, f"{f['time_slot']} (early/late slots are skipped more often)"))

    return _clip(base + du + ns + spec + slot), factors


def score_ecommerce(f):
    """f: payment_method, amount, category, order_age_days."""
    factors = []
    base = 10
    cod = 35 if f["payment_method"] == "COD" else 0
    factors.append(("Payment method", cod,
                    f"{f['payment_method']} "
                    f"({'cash-on-delivery is the biggest refusal driver' if cod else 'prepaid is low risk'})"))

    amt = (f["amount"] - 5000) / 45000 * 14
    factors.append(("Order value", amt, f"Rs {f['amount']:,}"))

    cat = {"Electronics": 10, "Clothing": 6, "Home": 3, "Beauty": 2}[f["category"]]
    factors.append(("Product category", cat, f"{f['category']} (return/refusal propensity)"))

    age = f["order_age_days"] * 0.8
    factors.append(("Order age", age, f"{f['order_age_days']} day(s) unconfirmed"))

    return _clip(base + cod + amt + cat + age), factors


def score_banking(f):
    """f: loan_amount, installment, loan_type, days_until_payment, prev_late."""
    factors = []
    base = 14
    late = f["prev_late"] * 12
    factors.append(("Repayment history", late,
                    f"{f['prev_late']} late payment(s) in last 12 months "
                    f"({'strong default predictor' if f['prev_late'] else 'clean record'})"))

    burden = (f["installment"] - 25000) / 175000 * 26
    factors.append(("Installment burden", burden, f"Rs {f['installment']:,}/month"))

    lt = {"Personal": 14, "Business": 10, "Auto": 6, "Home": 2}[f["loan_type"]]
    factors.append(("Loan type", lt, f"{f['loan_type']} ({'unsecured' if f['loan_type'] in ('Personal', 'Business') else 'secured'})"))

    due = (30 - f["days_until_payment"]) / 30 * 8
    factors.append(("Payment proximity", due, f"{f['days_until_payment']} day(s) to next payment"))

    amt = (f["loan_amount"] - 500000) / 4500000 * 8
    factors.append(("Loan size", amt, f"Rs {f['loan_amount']:,}"))

    return _clip(base + late + burden + lt + due + amt), factors


def score_dining(f):
    """f: distance, amount, repeat_customer, payment_method."""
    factors = []
    base = 14
    cod = 22 if f["payment_method"] == "COD" else 0
    factors.append(("Payment method", cod,
                    f"{f['payment_method']} ({'cash-on-delivery food orders are refused more' if cod else 'prepaid'})"))

    dist = f["distance"] * 2.4
    factors.append(("Delivery distance", dist, f"{f['distance']} km ({'long routes flake more' if f['distance'] >= 8 else 'short hop'})"))

    if f["amount"] < 4000:
        small = 12
        small_note = f"Rs {f['amount']:,} (small orders flake more)"
    else:
        small = f["amount"] / 15000 * 4
        small_note = f"Rs {f['amount']:,}"
    factors.append(("Order value", small, small_note))

    repeat = -6 if f["repeat_customer"] == "Yes" else 14
    factors.append(("Customer history", repeat,
                    f"{'Repeat customer' if f['repeat_customer'] == 'Yes' else 'First-time customer'}"))

    return _clip(base + cod + dist + small + repeat), factors


SCORERS = {
    "Hospitality": score_hospitality,
    "Healthcare": score_healthcare,
    "E-commerce": score_ecommerce,
    "Banking": score_banking,
    "Dining & Delivery": score_dining,
}


def features_from_row(industry, row):
    """Reconstruct the engine's feature dict from a dataframe row."""
    if industry == "Hospitality":
        return {"lead_days": int(row["Lead Time (days)"]), "duration": int(row["Duration"]),
                "amount": int(row["Amount (Rs)"]), "room_type": row["Room Type"],
                "prev_cancel": int(row["Previous Cancellations"])}
    if industry == "Healthcare":
        return {"days_until": int(row["Days Until Appointment"]),
                "prev_no_shows": int(row["Previous No-shows"]),
                "specialist": row["Specialist"], "time_slot": row["Time Slot"]}
    if industry == "E-commerce":
        return {"payment_method": row["Payment Method"], "amount": int(row["Amount (Rs)"]),
                "category": row["Product Category"], "order_age_days": int(row["Order Age (days)"])}
    if industry == "Banking":
        return {"loan_amount": int(row["Loan Amount (Rs)"]), "installment": int(row["Monthly Installment (Rs)"]),
                "loan_type": row["Loan Type"], "days_until_payment": int(row["Days Until Next Payment"]),
                "prev_late": int(row["Late Payments (12mo)"])}
    if industry == "Dining & Delivery":
        return {"distance": float(row["Delivery Distance (km)"]), "amount": int(row["Amount (Rs)"]),
                "repeat_customer": row["Repeat Customer"], "payment_method": row["Payment Method"]}
    return {}


def explain_row(industry, row):
    """Return the (score, factors) that produced this row's risk score."""
    return SCORERS[industry](features_from_row(industry, row))


def get_risk_level(score):
    if score >= 70:
        return "High Risk", RED
    elif score >= 40:
        return "Medium Risk", ORANGE
    return "Low Risk", GREEN


def get_recommended_action(score):
    if score >= 70:
        return "🔒 SECURE", "Request deposit/prepayment or OTP confirmation"
    elif score >= 40:
        return "🔔 NUDGE", "Send automated SMS/WhatsApp reminder"
    return "✅ CONFIRM", "No action needed"


def _status_for(industry, score):
    if industry == "Banking":
        return "Default Risk" if score >= 70 else ("At Risk" if score >= 40 else "Active")
    if industry == "E-commerce":
        return "At Risk" if score >= 70 else ("Processing" if score >= 40 else "In Transit")
    if industry == "Dining & Delivery":
        return "At Risk" if score >= 70 else ("Preparing" if score >= 40 else "In Delivery")
    # Hospitality + Healthcare
    return "At Risk" if score >= 70 else ("Pending" if score >= 40 else "Confirmed")


# ===========================================================================
# 2. DATA GENERATION  (seeded + feature-driven scores)
# ===========================================================================
def generate_mock_data(seed=42, n=20):
    """Generate reproducible mock data whose Risk Score is computed by the
    engine from each row's own features."""
    rng = np.random.default_rng(seed)
    now = datetime.now()
    data = {}

    # --- Hospitality -------------------------------------------------------
    lead = rng.integers(1, 31, n)
    rows = pd.DataFrame({
        "Commitment ID": [f"HOS-{i:04d}" for i in range(1, n + 1)],
        "Customer Name": [f"Guest {i}" for i in range(1, n + 1)],
        "Booking Date": [now - timedelta(days=int(x)) for x in rng.integers(0, 60, n)],
        "Lead Time (days)": lead,
        "Check-in Date": [now + timedelta(days=int(x)) for x in lead],
        "Room Type": rng.choice(["Deluxe", "Standard", "Suite"], n),
        "Duration": rng.integers(1, 11, n),
        "Previous Cancellations": rng.integers(0, 4, n),
        "Amount (Rs)": rng.integers(15000, 100000, n),
    })
    data["Hospitality"] = _apply_scores(rows, "Hospitality")

    # --- Healthcare --------------------------------------------------------
    du = rng.integers(1, 46, n)
    rows = pd.DataFrame({
        "Commitment ID": [f"HLC-{i:04d}" for i in range(1, n + 1)],
        "Patient Name": [f"Patient {i}" for i in range(1, n + 1)],
        "Appointment Date": [now + timedelta(days=int(x)) for x in du],
        "Days Until Appointment": du,
        "Specialist": rng.choice(["Cardiology", "Orthopedics", "Neurology", "Dentistry"], n),
        "Time Slot": rng.choice(["9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM"], n),
        "Previous No-shows": rng.integers(0, 4, n),
    })
    data["Healthcare"] = _apply_scores(rows, "Healthcare")

    # --- E-commerce --------------------------------------------------------
    age = rng.integers(0, 14, n)
    rows = pd.DataFrame({
        "Commitment ID": [f"ECM-{i:04d}" for i in range(1, n + 1)],
        "Order ID": [f"ORD-{i:05d}" for i in range(50001, 50001 + n)],
        "Customer": [f"Customer {i}" for i in range(1, n + 1)],
        "Order Date": [now - timedelta(days=int(x)) for x in age],
        "Order Age (days)": age,
        "Amount (Rs)": rng.integers(5000, 50000, n),
        "Payment Method": rng.choice(["COD", "Prepaid"], n),
        "Product Category": rng.choice(["Electronics", "Clothing", "Home", "Beauty"], n),
    })
    data["E-commerce"] = _apply_scores(rows, "E-commerce")

    # --- Banking -----------------------------------------------------------
    rows = pd.DataFrame({
        "Commitment ID": [f"BNK-{i:04d}" for i in range(1, n + 1)],
        "Loan ID": [f"LN-{i:06d}" for i in range(100001, 100001 + n)],
        "Customer Name": [f"Borrower {i}" for i in range(1, n + 1)],
        "Loan Amount (Rs)": rng.integers(500000, 5000000, n),
        "Loan Type": rng.choice(["Personal", "Auto", "Business", "Home"], n),
        "Monthly Installment (Rs)": rng.integers(25000, 200000, n),
        "Days Until Next Payment": rng.integers(0, 30, n),
        "Late Payments (12mo)": rng.integers(0, 4, n),
    })
    data["Banking"] = _apply_scores(rows, "Banking")

    # --- Dining & Delivery -------------------------------------------------
    rows = pd.DataFrame({
        "Commitment ID": [f"DNI-{i:04d}" for i in range(1, n + 1)],
        "Order ID": [f"FOOD-{i:05d}" for i in range(30001, 30001 + n)],
        "Restaurant": rng.choice(["Italian Bistro", "Thai Spice", "Burger Palace", "Sushi House"], n),
        "Order Time": [now - timedelta(minutes=int(x)) for x in rng.integers(5, 120, n)],
        "Amount (Rs)": rng.integers(2000, 15000, n),
        "Delivery Distance (km)": np.round(rng.uniform(1, 15, n), 1),
        "Repeat Customer": rng.choice(["Yes", "No"], n),
        "Payment Method": rng.choice(["COD", "Prepaid"], n),
    })
    data["Dining & Delivery"] = _apply_scores(rows, "Dining & Delivery")

    return data


def _apply_scores(df, industry):
    """Compute Risk Score for every row via the engine, then derive Status."""
    scores = [SCORERS[industry](features_from_row(industry, r))[0] for _, r in df.iterrows()]
    df["Risk Score"] = scores
    df["Status"] = [_status_for(industry, s) for s in scores]
    return df


# ===========================================================================
# 3. STREAMLIT UI
# ===========================================================================
def _inject_style():
    st.markdown("""
        <style>
        .main-header {
            background: linear-gradient(135deg, #003D5B 0%, #00B8A9 100%);
            padding: 30px; border-radius: 10px; color: white; margin-bottom: 20px;
        }
        .metric-card {
            background: white; padding: 20px; border-radius: 8px;
            border-left: 4px solid #00B8A9; box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .metric-card h4 { color: #003D5B; margin: 0; }
        .risk-high   { background: rgba(238, 90, 111, 0.10); border-left-color: #EE5A6F; }
        .risk-medium { background: rgba(247, 160, 114, 0.10); border-left-color: #F7A072; }
        .risk-low    { background: rgba(38, 196, 133, 0.10); border-left-color: #26C485; }
        .notification-badge {
            background: #EE5A6F; color: white; padding: 2px 8px;
            border-radius: 12px; font-size: 12px; font-weight: bold;
        }
        </style>
    """, unsafe_allow_html=True)


def _amount_col(df):
    for c in ("Amount (Rs)", "Loan Amount (Rs)"):
        if c in df.columns:
            return c
    return None


def gauge(score, title="Risk Score"):
    level, color = get_risk_level(score)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        title={"text": f"{title} – {level}"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar": {"color": color},
            "steps": [
                {"range": [0, 40], "color": "rgba(38,196,133,0.25)"},
                {"range": [40, 70], "color": "rgba(247,160,114,0.25)"},
                {"range": [70, 100], "color": "rgba(238,90,111,0.25)"},
            ],
        },
    ))
    fig.update_layout(height=280, margin=dict(t=60, b=10, l=20, r=20))
    return fig


def factor_chart(factors):
    fac = pd.DataFrame(factors, columns=["Factor", "Points", "Note"])
    fac = fac.sort_values("Points")
    fig = px.bar(fac, x="Points", y="Factor", orientation="h", text="Points",
                 color="Points",
                 color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]])
    fig.update_traces(texttemplate="%{text:+.0f}")
    fig.update_layout(height=300, showlegend=False, xaxis_title="Points added to risk",
                      yaxis_title="", coloraxis_showscale=False,
                      margin=dict(t=10, b=10))
    return fig


# ----------------------------------- pages ---------------------------------
def show_executive_overview(all_data, threshold_alert=70, show_notifications=True):
    st.subheader("🎯 Real-Time Risk Dashboard")

    total_commitments = sum(len(df) for df in all_data.values())
    total_high_risk = sum(len(df[df["Risk Score"] >= threshold_alert]) for df in all_data.values())
    total_losses = sum(
        df[df["Risk Score"] >= 70]["Amount (Rs)"].sum() * 0.8 if "Amount (Rs)" in df.columns
        else len(df[df["Risk Score"] >= 70]) * 50000
        for df in all_data.values()
    )
    recovery_rate = 22

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h4>Commitments Scored</h4>"
                f"<p style='font-size:28px;color:#00B8A9;margin:10px 0;'>{total_commitments:,}</p>"
                f"<p style='color:#666;font-size:12px;'>Across all industries</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card risk-high'><h4>Flagged High-Risk</h4>"
                f"<p style='font-size:28px;color:#EE5A6F;margin:10px 0;'>{total_high_risk}</p>"
                f"<p style='color:#666;font-size:12px;'>At/above threshold</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card'><h4>Losses Intercepted</h4>"
                f"<p style='font-size:28px;color:#00B8A9;margin:10px 0;'>Rs {total_losses/1_000_000:.1f}M</p>"
                f"<p style='color:#666;font-size:12px;'>Prevented this month</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card'><h4>Recovery Rate</h4>"
                f"<p style='font-size:28px;color:#26C485;margin:10px 0;'>{recovery_rate}%</p>"
                f"<p style='color:#666;font-size:12px;'>Conversion to firm bookings</p></div>", unsafe_allow_html=True)

    if show_notifications:
        st.markdown("---")
        st.subheader("🔔 Critical Alerts")
        alerts = []
        for industry, df in all_data.items():
            for _, row in df[df["Risk Score"] >= 75].head(2).iterrows():
                amt_col = _amount_col(df)
                alerts.append({"Industry": industry, "ID": row["Commitment ID"],
                               "Risk Score": row["Risk Score"],
                               "Amount": row[amt_col] if amt_col else "N/A"})
        if alerts:
            for row in alerts[:5]:
                a1, a2, a3, a4 = st.columns([1, 2, 1.5, 1.5])
                a1.markdown(f"<div class='notification-badge'>⚠️ {row['Risk Score']}</div>", unsafe_allow_html=True)
                a2.write(f"**{row['Industry']}** – {row['ID']}")
                action, _ = get_recommended_action(row["Risk Score"])
                a3.write(f"*{action}*")
                a4.write(f"Rs {row['Amount']:,}" if isinstance(row["Amount"], (int, np.integer)) else row["Amount"])
        else:
            st.success("No commitments above the alert threshold right now.")

    st.markdown("---")
    st.subheader("📊 Industry Performance Summary")
    metrics = [{
        "Industry": industry, "Commitments": len(df),
        "High Risk": int((df["Risk Score"] >= 70).sum()),
        "Avg Risk Score": round(df["Risk Score"].mean(), 1),
        "Action Items": int((df["Risk Score"] >= 40).sum()),
    } for industry, df in all_data.items()]
    idf = pd.DataFrame(metrics)

    g1, g2 = st.columns(2)
    with g1:
        fig = px.bar(idf, x="Industry", y=["High Risk", "Action Items"], barmode="group",
                     color_discrete_map={"High Risk": RED, "Action Items": ORANGE})
        fig.update_layout(title="Risk Items by Industry", xaxis_title="", yaxis_title="Count",
                          hovermode="x unified", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with g2:
        fig = px.bar(idf.sort_values("Avg Risk Score", ascending=False), x="Industry", y="Avg Risk Score",
                     color="Avg Risk Score", color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]])
        fig.update_layout(title="Average Risk Score by Industry", xaxis_title="",
                          yaxis_title="Average Risk Score", height=350)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📈 Risk Distribution")
    all_scores = pd.concat([df["Risk Score"] for df in all_data.values()]).reset_index(drop=True)
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=all_scores, nbinsx=20, marker_color=PRIMARY))
    fig.add_vline(x=40, line_dash="dash", line_color=ORANGE, annotation_text="Medium")
    fig.add_vline(x=70, line_dash="dash", line_color=RED, annotation_text="High")
    fig.update_layout(title="System-Wide Risk Score Distribution", xaxis_title="Risk Score",
                      yaxis_title="Number of Commitments", height=400, showlegend=False)
    st.plotly_chart(fig, use_container_width=True)


def show_industry_dashboard(df, industry_name, threshold_alert=70):
    st.subheader(f"Dashboard: {industry_name}")

    high = int((df["Risk Score"] >= 70).sum())
    med = int(((df["Risk Score"] >= 40) & (df["Risk Score"] < 70)).sum())
    low = int((df["Risk Score"] < 40).sum())

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<div class='metric-card'><h4>Total Commitments</h4>"
                f"<p style='font-size:28px;color:#00B8A9;margin:10px 0;'>{len(df)}</p></div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='metric-card risk-high'><h4>High Risk</h4>"
                f"<p style='font-size:28px;color:#EE5A6F;margin:10px 0;'>{high}</p></div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='metric-card risk-medium'><h4>Medium Risk</h4>"
                f"<p style='font-size:28px;color:#F7A072;margin:10px 0;'>{med}</p></div>", unsafe_allow_html=True)
    c4.markdown(f"<div class='metric-card risk-low'><h4>Low Risk</h4>"
                f"<p style='font-size:28px;color:#26C485;margin:10px 0;'>{low}</p></div>", unsafe_allow_html=True)

    st.markdown("---")
    p1, p2 = st.columns(2)
    with p1:
        fig = go.Figure(data=[go.Pie(
            labels=["High Risk (≥70)", "Medium Risk (40-69)", "Low Risk (<40)"],
            values=[high, med, low], marker=dict(colors=[RED, ORANGE, GREEN]),
            hovertemplate="<b>%{label}</b><br>Count: %{value}<br>%{percent}<extra></extra>")])
        fig.update_layout(title="Risk Distribution", height=350)
        st.plotly_chart(fig, use_container_width=True)
    with p2:
        top = df.sort_values("Risk Score", ascending=False).head(10)
        fig = px.bar(top, x="Risk Score", y="Commitment ID", orientation="h", color="Risk Score",
                     color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]])
        fig.update_layout(title="Top 10 Highest Risk Commitments", xaxis_title="Risk Score",
                          yaxis_title="", height=350, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("📋 Detailed Commitment List")
    f1, f2, f3 = st.columns(3)
    risk_filter = f1.selectbox("Filter by Risk Level",
                               ["All", "High Risk (≥70)", "Medium Risk (40-69)", "Low Risk (<40)"])
    sort_by = f2.selectbox("Sort by",
                           ["Risk Score (High to Low)", "Risk Score (Low to High)", "Commitment ID"])
    items_to_show = f3.slider("Show items", 5, len(df), min(10, len(df)))

    fdf = df.copy()
    if risk_filter == "High Risk (≥70)":
        fdf = fdf[fdf["Risk Score"] >= 70]
    elif risk_filter == "Medium Risk (40-69)":
        fdf = fdf[(fdf["Risk Score"] >= 40) & (fdf["Risk Score"] < 70)]
    elif risk_filter == "Low Risk (<40)":
        fdf = fdf[fdf["Risk Score"] < 40]

    if sort_by == "Risk Score (High to Low)":
        fdf = fdf.sort_values("Risk Score", ascending=False)
    elif sort_by == "Risk Score (Low to High)":
        fdf = fdf.sort_values("Risk Score", ascending=True)
    else:
        fdf = fdf.sort_values("Commitment ID")

    st.download_button("⬇️ Download this view (CSV)",
                       fdf.to_csv(index=False).encode("utf-8"),
                       file_name=f"confirmiq_{industry_name.lower().replace(' ', '_')}.csv",
                       mime="text/csv")

    for _, row in fdf.head(items_to_show).iterrows():
        risk_level, _ = get_risk_level(row["Risk Score"])
        action, description = get_recommended_action(row["Risk Score"])
        with st.expander(f"🔹 {row['Commitment ID']} – Risk: {row['Risk Score']}/100 – {risk_level}",
                         expanded=(row["Risk Score"] >= 70)):
            d1, d2, d3 = st.columns(3)
            with d1:
                st.metric("Risk Score", f"{row['Risk Score']}/100")
                st.write(f"**Status:** {row['Status']}")
            with d2:
                st.write("**Recommended Action:**")
                st.markdown(f"### {action}")
                st.write(f"*{description}*")
            with d3:
                hide = {"Commitment ID", "Risk Score", "Status"}
                for col in row.index:
                    if col in hide:
                        continue
                    value = row[col]
                    if hasattr(value, "strftime"):
                        value = value.strftime("%Y-%m-%d")
                    st.write(f"**{col}:** {value}")

            st.divider()
            st.write("**Why this score?** (actual factors that produced it)")
            _, factors = explain_row(industry_name, row)
            for factor, points, note in sorted(factors, key=lambda x: -x[1]):
                sign = "🔺" if points > 0 else ("🔻" if points < 0 else "▪️")
                st.write(f"{sign} **{factor}** ({points:+.0f}) — {note}")


def show_score_commitment():
    st.subheader("🧮 Score a New Commitment")
    st.caption("Enter a commitment's details to get a live risk score, recommended action and factor breakdown.")
    industry = st.selectbox("Industry", INDUSTRIES)

    if industry == "Hospitality":
        c1, c2 = st.columns(2)
        f = {
            "lead_days": c1.slider("Days until check-in", 1, 30, 14),
            "duration": c1.slider("Nights", 1, 10, 2),
            "amount": c2.slider("Booking value (Rs)", 15000, 100000, 45000, step=1000),
            "room_type": c2.selectbox("Room type", ["Standard", "Deluxe", "Suite"]),
            "prev_cancel": c1.slider("Previous cancellations", 0, 3, 0),
        }
    elif industry == "Healthcare":
        c1, c2 = st.columns(2)
        f = {
            "days_until": c1.slider("Days until appointment", 1, 45, 14),
            "prev_no_shows": c1.slider("Previous no-shows", 0, 3, 0),
            "specialist": c2.selectbox("Specialist", ["Cardiology", "Orthopedics", "Neurology", "Dentistry"]),
            "time_slot": c2.selectbox("Time slot", ["9:00 AM", "11:00 AM", "2:00 PM", "4:00 PM"]),
        }
    elif industry == "E-commerce":
        c1, c2 = st.columns(2)
        f = {
            "payment_method": c1.selectbox("Payment method", ["Prepaid", "COD"]),
            "amount": c2.slider("Order value (Rs)", 5000, 50000, 20000, step=500),
            "category": c1.selectbox("Category", ["Electronics", "Clothing", "Home", "Beauty"]),
            "order_age_days": c2.slider("Days unconfirmed", 0, 14, 2),
        }
    elif industry == "Banking":
        c1, c2 = st.columns(2)
        f = {
            "loan_amount": c1.slider("Loan amount (Rs)", 500000, 5000000, 1500000, step=50000),
            "installment": c2.slider("Monthly installment (Rs)", 25000, 200000, 75000, step=5000),
            "loan_type": c1.selectbox("Loan type", ["Personal", "Auto", "Business", "Home"]),
            "days_until_payment": c2.slider("Days to next payment", 0, 30, 10),
            "prev_late": c1.slider("Late payments (last 12 months)", 0, 3, 0),
        }
    else:  # Dining & Delivery
        c1, c2 = st.columns(2)
        f = {
            "distance": c1.slider("Delivery distance (km)", 1.0, 15.0, 5.0, step=0.5),
            "amount": c2.slider("Order value (Rs)", 2000, 15000, 5000, step=250),
            "repeat_customer": c1.selectbox("Repeat customer?", ["Yes", "No"]),
            "payment_method": c2.selectbox("Payment method", ["Prepaid", "COD"]),
        }

    score, factors = SCORERS[industry](f)
    action, description = get_recommended_action(score)

    st.markdown("---")
    r1, r2 = st.columns([1, 1])
    with r1:
        st.plotly_chart(gauge(score), use_container_width=True)
        st.markdown(f"### {action}")
        st.write(f"*{description}*")
    with r2:
        st.write("**Factor breakdown**")
        st.plotly_chart(factor_chart(factors), use_container_width=True)


def show_analytics(all_data):
    st.subheader("📊 Advanced Analytics & Insights")

    parts = []
    for industry, df in all_data.items():
        d = df.copy()
        d["Industry"] = industry
        parts.append(d)
    combined = pd.concat(parts, ignore_index=True)

    st.markdown("---")
    st.subheader("💡 Key Insights")
    k1, k2, k3 = st.columns(3)
    by_ind = combined.groupby("Industry")["Risk Score"].mean()
    k1.info(f"**Highest Risk Industry**\n\n{by_ind.idxmax()} averages {by_ind.max():.1f}/100.")

    exposure = 0
    for industry, df in all_data.items():
        col = _amount_col(df)
        if col:
            exposure += df[df["Risk Score"] >= 70][col].sum()
    k2.warning(f"**Financial Exposure**\n\nHigh-risk commitment value: Rs {exposure/1_000_000:.1f}M")

    recoverable = int(((combined["Risk Score"] >= 40) & (combined["Risk Score"] < 70)).sum())
    k3.success(f"**Recovery Opportunity**\n\n{recoverable} medium-risk commitments respond to NUDGE actions.")

    st.markdown("---")
    a1, a2 = st.columns(2)
    with a1:
        ir = combined.groupby("Industry")["Risk Score"].agg(["mean", "count"]).reset_index()
        fig = px.scatter(ir, x="count", y="mean", size="count", color="mean", text="Industry",
                         color_continuous_scale=[[0, GREEN], [0.5, ORANGE], [1, RED]],
                         title="Industry Volume vs Average Risk")
        fig.update_traces(textposition="top center")
        fig.update_layout(xaxis_title="Number of Commitments", yaxis_title="Average Risk Score", height=400)
        st.plotly_chart(fig, use_container_width=True)
    with a2:
        combined["Recommended_Action"] = combined["Risk Score"].apply(
            lambda x: "🔒 SECURE" if x >= 70 else ("🔔 NUDGE" if x >= 40 else "✅ CONFIRM"))
        counts = combined["Recommended_Action"].value_counts()
        fig = px.pie(values=counts.values, names=counts.index, color=counts.index,
                     color_discrete_map={"🔒 SECURE": RED, "🔔 NUDGE": ORANGE, "✅ CONFIRM": GREEN},
                     title="Distribution of Recommended Actions")
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("🎯 Recovery Potential Analysis")
    recovery = []
    for industry, df in all_data.items():
        med = df[(df["Risk Score"] >= 40) & (df["Risk Score"] < 70)]
        if len(med):
            col = _amount_col(df)
            val = med[col].sum() if col else len(med) * 50000
            recovery.append({"Industry": industry, "Medium Risk Items": len(med),
                             "Potential Recovery (Rs)": int(val), "Recovery Rate": "15-20%"})
    rdf = pd.DataFrame(recovery)

    if not rdf.empty:
        rc1, rc2 = st.columns(2)
        with rc1:
            fig = px.bar(rdf, x="Industry", y="Medium Risk Items", color="Medium Risk Items",
                         color_continuous_scale=[[0, "#FFE5E5"], [1, RED]], title="NUDGE Candidates by Industry")
            fig.update_layout(xaxis_title="", yaxis_title="Items", height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
        with rc2:
            fig = px.bar(rdf, x="Industry", y="Potential Recovery (Rs)", color="Potential Recovery (Rs)",
                         color_continuous_scale=[[0, "#E5F5F0"], [1, PRIMARY]], title="Potential Recovery Value")
            fig.update_layout(xaxis_title="", yaxis_title="Recovery Value (Rs)", height=350, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("💰 Business Model Impact")
    b1, b2, b3 = st.columns(3)
    total = len(combined)
    b1.metric("Monthly SaaS Revenue", f"Rs {5*50000:,}", f"{int(total/100)} commitments")
    b2.metric("API Transaction Revenue", f"Rs {total*5:,}", f"{total} transactions")
    rev_share = rdf["Potential Recovery (Rs)"].sum() * 0.18 if not rdf.empty else 0
    b3.metric("Revenue Share (18%)", f"Rs {int(rev_share):,}", "From recovered value")

    st.markdown("---")
    st.download_button("⬇️ Download all commitments (CSV)",
                       combined.to_csv(index=False).encode("utf-8"),
                       file_name="confirmiq_all_commitments.csv", mime="text/csv")


# ===========================================================================
# 4. MAIN
# ===========================================================================
@st.cache_data
def _cached_data(seed):
    return generate_mock_data(seed)


def main():
    st.set_page_config(page_title="ConfirmIQ - Commitment Risk Engine",
                       page_icon="🔐", layout="wide", initial_sidebar_state="expanded")
    _inject_style()

    h1, h2 = st.columns([3, 1])
    h1.markdown("""
        <div class="main-header">
            <h1>🔐 ConfirmIQ</h1>
            <h3>The Commitment Risk Engine</h3>
            <p>Predicting which bookings, appointments, orders and repayments will break —
            before they cost Sri Lankan businesses money.</p>
        </div>""", unsafe_allow_html=True)
    h2.markdown(f"""
        <div style='text-align:right;margin-top:20px;'>
            <p><strong>Last Updated:</strong></p>
            <p style='color:#00B8A9;font-size:14px;'>{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>""", unsafe_allow_html=True)

    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.radio("Select Dashboard View",
                            ["📊 Executive Overview", "🧮 Score a Commitment",
                             "🏨 Hospitality", "🏥 Healthcare", "📦 E-commerce",
                             "🏦 Banking", "🍽️ Dining & Delivery", "📈 Analytics & Insights"])

    st.sidebar.markdown("---")
    st.sidebar.title("⚙️ Settings")
    show_notifications = st.sidebar.checkbox("Show Notifications", value=True)
    threshold_alert = st.sidebar.slider("Risk Alert Threshold", 0, 100, 70)
    seed = st.sidebar.number_input("Sample seed", 0, 9999, 42, step=1,
                                   help="Change to regenerate a different reproducible sample.")

    st.sidebar.markdown("---")
    st.sidebar.info("""**ConfirmIQ Features:**
- Feature-based risk scoring (0–100)
- Multi-industry support
- Live single-commitment scorer
- Explainable, per-row factor breakdowns
- CSV export""")

    all_data = _cached_data(int(seed))

    if page == "📊 Executive Overview":
        show_executive_overview(all_data, threshold_alert, show_notifications)
    elif page == "🧮 Score a Commitment":
        show_score_commitment()
    elif page == "📈 Analytics & Insights":
        show_analytics(all_data)
    else:
        name = page.split(" ", 1)[1]
        show_industry_dashboard(all_data[name], name, threshold_alert)


if __name__ == "__main__":
    main()
