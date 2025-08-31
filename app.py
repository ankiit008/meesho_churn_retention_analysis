# app.py — Customer Retention & Churn Analysis (robust build)

import os
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# ML
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

# Optional warehouse deps (import gently so CSV mode still works)
try:
    from sqlalchemy import create_engine, text as sa_text
except Exception:
    create_engine = None
    sa_text = None

try:
    from pandas_gbq import read_gbq as gbq_read
except Exception:
    gbq_read = None


APP_DIR = Path(__file__).parent
st.set_page_config(page_title="Customer Retention & Churn", layout="wide")
st.title("🧲 Customer Retention & Churn Analysis")
st.caption(
    "Cohort retention tables + RFM features + churn prediction. "
    "Supports CSV, uploads, Postgres, and BigQuery."
)

# ------------------------ Data loaders ------------------------

@st.cache_data
def load_local_csv(name: str) -> pd.DataFrame:
    path = APP_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)

@st.cache_data
def load_postgres(db_url: str):
    if create_engine is None:
        raise RuntimeError("SQLAlchemy not installed. Add to requirements: sqlalchemy psycopg2-binary")
    engine = create_engine(db_url, pool_pre_ping=True)
    with engine.begin() as conn:
        cust = pd.read_sql(sa_text("SELECT * FROM customers"), conn)
        orders = pd.read_sql(sa_text("SELECT * FROM orders"), conn)
        sess = pd.read_sql(sa_text("SELECT * FROM sessions"), conn)
    return cust, orders, sess

@st.cache_data
def load_bigquery(project: str, dataset: str, t_c="customers", t_o="orders", t_s="sessions"):
    if gbq_read is None:
        raise RuntimeError("pandas-gbq not installed. Add to requirements: pandas-gbq google-cloud-bigquery")
    q = lambda t: f"SELECT * FROM `{project}.{dataset}.{t}`"
    cust = gbq_read(q(t_c), project_id=project)
    orders = gbq_read(q(t_o), project_id=project)
    sess = gbq_read(q(t_s), project_id=project)
    return cust, orders, sess

# ------------------------ Sidebar source ------------------------

st.sidebar.header("Data Source")
source = st.sidebar.selectbox(
    "Choose dataset", ["Sample (CSV)", "Upload CSVs", "Postgres", "BigQuery"], index=0
)

customers = orders = sessions = None

if source == "Sample (CSV)":
    customers = load_local_csv("customers.csv")
    orders    = load_local_csv("orders.csv")
    sessions  = load_local_csv("sessions.csv")

elif source == "Upload CSVs":
    c = st.sidebar.file_uploader("customers.csv", type=["csv"])
    o = st.sidebar.file_uploader("orders.csv", type=["csv"])
    s = st.sidebar.file_uploader("sessions.csv", type=["csv"])
    if c is not None and o is not None and s is not None:
        customers = pd.read_csv(c)
        orders    = pd.read_csv(o)
        sessions  = pd.read_csv(s)
    else:
        st.stop()

elif source == "Postgres":
    st.sidebar.caption("Paste DATABASE_URL (or put in Secrets). e.g. postgresql+psycopg2://user:pass@host:5432/db")
    db_url = os.getenv("DATABASE_URL", "")
    db_url = st.sidebar.text_input("DATABASE_URL", db_url, type="password")
    if not db_url:
        st.warning("Enter DATABASE_URL to load data.")
        st.stop()
    try:
        customers, orders, sessions = load_postgres(db_url)
    except Exception as e:
        st.error(f"Postgres load error: {e}")
        st.stop()

elif source == "BigQuery":
    st.sidebar.caption("Provide project & dataset (auth via service account JSON in Secrets).")
    project = st.sidebar.text_input("BQ project", os.getenv("BQ_PROJECT", ""))
    dataset = st.sidebar.text_input("BQ dataset", os.getenv("BQ_DATASET", ""))
    t_c = st.sidebar.text_input("Customers table", os.getenv("BQ_TABLE_CUST", "customers"))
    t_o = st.sidebar.text_input("Orders table", os.getenv("BQ_TABLE_ORD", "orders"))
    t_s = st.sidebar.text_input("Sessions table", os.getenv("BQ_TABLE_SESS", "sessions"))
    if not project or not dataset:
        st.warning("Enter BigQuery project and dataset to proceed.")
        st.stop()
    try:
        customers, orders, sessions = load_bigquery(project, dataset, t_c, t_o, t_s)
    except Exception as e:
        st.error(f"BigQuery load error: {e}")
        st.stop()

if customers is None or orders is None or sessions is None:
    st.warning("Provide all three datasets (customers, orders, sessions).")
    st.stop()

# ------------------------ Normalize and parse dates ------------------------

def to_dt(df, col):
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
    return df

customers = to_dt(customers, "signup_date")
orders    = to_dt(orders, "order_date")
sessions  = to_dt(sessions, "session_date")

# ------------------------ Filters ------------------------

def _date_range_defaults():
    if not orders.empty and "order_date" in orders.columns:
        return orders["order_date"].min().date(), orders["order_date"].max().date()
    if not customers.empty and "signup_date" in customers.columns:
        return customers["signup_date"].min().date(), customers["signup_date"].max().date()
    today = pd.Timestamp.today().date()
    return today, today

min_d, max_d = _date_range_defaults()
window = st.sidebar.date_input("Order window", (min_d, max_d), min_value=min_d, max_value=max_d)

tiers_all = sorted([x for x in customers.get("city_tier", pd.Series(dtype=str)).dropna().unique()])
chans_all = sorted([x for x in customers.get("acq_channel", pd.Series(dtype=str)).dropna().unique()])

tiers = st.sidebar.multiselect("City tier", tiers_all, default=tiers_all)
chans = st.sidebar.multiselect("Acq channel", chans_all, default=chans_all)

if "city_tier" in customers.columns and tiers:
    customers = customers[customers["city_tier"].isin(tiers)]
if "acq_channel" in customers.columns and chans:
    customers = customers[customers["acq_channel"].isin(chans)]

start_date = pd.to_datetime(window[0])
end_date   = pd.to_datetime(window[1])

if not orders.empty and "order_date" in orders.columns:
    orders = orders[orders["order_date"].between(start_date, end_date)]
if not sessions.empty and "session_date" in sessions.columns:
    sessions = sessions[sessions["session_date"].between(start_date, end_date)]

# ------------------------ KPIs (robust) ------------------------

active_customers = orders["customer_id"].nunique() if "customer_id" in orders.columns else 0
total_customers  = customers["customer_id"].nunique() if "customer_id" in customers.columns else 0
gmv_sum = float(orders["gmv"].sum()) if "gmv" in orders.columns else 0.0

ref_date = end_date

# Build last_order_date safely
if not orders.empty and {"customer_id", "order_date"}.issubset(orders.columns):
    last = orders.groupby("customer_id")["order_date"].max().rename("last_order_date")
else:
    last = pd.Series(dtype="datetime64[ns]").rename("last_order_date")

cust = customers.copy()
if "customer_id" in cust.columns:
    cust = cust.merge(last, on="customer_id", how="left")
else:
    cust["last_order_date"] = pd.NaT

# Normalize odd cases
if "last_order_date" not in cust.columns:
    if "order_date" in cust.columns:
        cust = cust.rename(columns={"order_date": "last_order_date"})
    elif 0 in cust.columns:
        cust = cust.rename(columns={0: "last_order_date"})
    else:
        cust["last_order_date"] = pd.NaT

cust["last_order_date"] = pd.to_datetime(cust["last_order_date"], errors="coerce")
cust["days_since_last"] = (ref_date - cust["last_order_date"]).dt.days
cust["days_since_last"] = cust["days_since_last"].fillna(9999)
cust["churn_label"]     = (cust["days_since_last"] > 60).astype(int)
churn_rate = float(cust["churn_label"].mean() * 100) if len(cust) else 0.0

c1, c2, c3, c4 = st.columns(4)
c1.metric("Total Customers", f"{total_customers:,}")
c2.metric("Active Customers", f"{active_customers:,}")
c3.metric("GMV in Window (₹)", f"{gmv_sum:,.0f}")
c4.metric("Churn Rate", f"{churn_rate:.1f}%")

st.divider()

# ------------------------ Cohort retention (final safe version) ------------------------

st.subheader("Cohort Retention (signup month × months since first order)")

if orders.empty or "customer_id" not in orders.columns or "order_date" not in orders.columns:
    st.info("No orders in the selected window. Adjust the date range or data source to see cohorts.")
else:
    if "signup_date" not in customers.columns:
        st.info("Missing signup_date in customers; cannot compute cohort. (Check your data schema.)")
    else:
        o = orders.merge(customers[["customer_id", "signup_date"]], on="customer_id", how="inner")
        o["cohort_month"] = o["signup_date"].dt.to_period("M").astype(str)
        o["order_month"]  = o["order_date"].dt.to_period("M").astype(str)

        first = o.groupby("customer_id")["order_month"].min().rename("first_order_month")
        o = o.merge(first, on="customer_id", how="left")

        # Months since first order — convert to numeric safely, drop bad rows, cast to int
        idx = (pd.PeriodIndex(o["order_month"], freq="M")
               - pd.PeriodIndex(o["first_order_month"], freq="M"))
        o["cohort_idx"] = pd.to_numeric(pd.Series(idx), errors="coerce")
        o = o[o["cohort_idx"].notna()].copy()
        o["cohort_idx"] = o["cohort_idx"].astype(int)

        ret = (o.groupby(["cohort_month", "cohort_idx"])["customer_id"]
                 .nunique()
                 .reset_index(name="active_users"))
        base = (o.groupby("cohort_month")["customer_id"]
                  .nunique()
                  .reset_index(name="cohort_size"))
        ret = ret.merge(base, on="cohort_month", how="left")
        ret["retention_rate"] = ret["active_users"] / ret["cohort_size"].replace({0: np.nan})

        pivot = (ret.pivot(index="cohort_month", columns="cohort_idx",
                           values="retention_rate")
                   .fillna(0)
                   .round(3))
        st.dataframe(pivot.style.background_gradient(cmap="Greens"), use_container_width=True)

st.divider()

# ------------------------ RFM + churn model ------------------------

st.subheader("Churn Prediction (Logistic Regression)")

if orders.empty or "customer_id" not in orders.columns or "order_date" not in orders.columns:
    st.info("Not enough orders to compute RFM features.")
else:
    rfm = orders.groupby("customer_id").agg(
        recency=("order_date", lambda s: (ref_date - s.max()).days),
        frequency=("order_id", "count") if "order_id" in orders.columns else ("order_date", "count"),
        monetary=("gmv", "sum") if "gmv" in orders.columns else ("order_date", "count"),
        avg_discount=("discount_pct", "mean") if "discount_pct" in orders.columns else ("order_date", "count"),
    ).reset_index()

    # sessions last 30 days
    if "session_date" in sessions.columns:
        sessions["is30"] = (sessions["session_date"] >= (ref_date - pd.Timedelta(days=30))).astype(int)
        sess = sessions.groupby("customer_id").agg(
            sessions_30=("is30", "sum"),
            minutes_30=("minutes", "sum") if "minutes" in sessions.columns else ("is30", "sum"),
        ).reset_index()
    else:
        sess = pd.DataFrame(columns=["customer_id", "sessions_30", "minutes_30"])

    feat = cust[["customer_id", "churn_label"]].copy()
    for col in ["city_tier", "acq_channel", "age"]:  # keep demographics if present
        if col in cust.columns:
            feat[col] = cust[col]

    feat = feat.merge(rfm, on="customer_id", how="left").merge(sess, on="customer_id", how="left")
    feat = feat.fillna({
        "recency": 9999, "frequency": 0, "monetary": 0, "avg_discount": 0,
        "sessions_30": 0, "minutes_30": 0
    })

    # One-hot encoding (only if columns exist)
    for cat in ["city_tier", "acq_channel"]:
        if cat in feat.columns:
            feat = pd.get_dummies(feat, columns=[cat], drop_first=True)

    if feat["churn_label"].nunique() < 2 or len(feat) < 50:
        st.info("Not enough data/class balance for training a model. (Need at least two classes and ~50+ rows.)")
    else:
        X = feat.drop(columns=["customer_id", "churn_label"])
        y = feat["churn_label"].astype(int)

        num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        if not num_cols:
            st.info("No numeric features available for the model.")
        else:
            scaler = StandardScaler()
            Xs = X.copy()
            Xs[num_cols] = scaler.fit_transform(X[num_cols])

            X_train, X_test, y_train, y_test = train_test_split(
                Xs, y, test_size=0.25, random_state=42, stratify=y
            )

            model = LogisticRegression(max_iter=1000)
            model.fit(X_train, y_train)
            proba = model.predict_proba(X_test)[:, 1]
            pred = (proba >= 0.5).astype(int)
            acc = accuracy_score(y_test, pred)
            auc = roc_auc_score(y_test, proba)

            c1, c2 = st.columns(2)
            c1.metric("Accuracy", f"{acc*100:.1f}%")
            c2.metric("ROC AUC", f"{auc:.3f}")

            coef = pd.DataFrame({"feature": X.columns, "coef": model.coef_[0]}).sort_values(
                "coef", key=lambda s: s.abs(), ascending=False
            ).head(12)
            st.write("Top features by absolute weight:")
            st.dataframe(coef, use_container_width=True)

            # score all customers
            X_all = feat.drop(columns=["customer_id", "churn_label"]).copy()
            X_all[num_cols] = scaler.transform(X_all[num_cols])
            feat["churn_score"] = model.predict_proba(X_all)[:, 1]

            at_risk = feat[feat["churn_score"] >= 0.70].sort_values(
                "churn_score", ascending=False
            )[
                ["customer_id", "churn_score", "recency", "frequency", "monetary", "sessions_30"]
            ]

            st.subheader("At-risk customers (score ≥ 0.70)")
            st.dataframe(at_risk.head(200), use_container_width=True)

            st.download_button(
                "Download churn scores",
                feat[["customer_id", "churn_label", "churn_score"]].to_csv(index=False).encode("utf-8"),
                file_name="churn_scores.csv",
                mime="text/csv",
            )

st.caption("Tip: Use at-risk scores to target re-engagement campaigns (e.g., coupons for high-GMV but high-risk users).")
