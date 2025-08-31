
import streamlit as st, pandas as pd, numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score

APP_DIR = Path(__file__).parent
st.set_page_config(page_title="Customer Retention & Churn", layout="wide")
st.title("🧲 Customer Retention & Churn Analysis")
st.caption("Cohort retention tables + RFM features + churn prediction.")

st.sidebar.header("Data Source")
src = st.sidebar.selectbox("Choose dataset", ["Sample (CSV)", "Upload CSVs"], index=0)

def load_csv(name): return pd.read_csv(APP_DIR / name)

if src=="Sample (CSV)":
    customers = load_csv("customers.csv")
    orders    = load_csv("orders.csv")
    sessions  = load_csv("sessions.csv")
else:
    c = st.sidebar.file_uploader("customers.csv", type=["csv"])
    o = st.sidebar.file_uploader("orders.csv", type=["csv"])
    s = st.sidebar.file_uploader("sessions.csv", type=["csv"])
    if not (c and o and s): st.stop()
    customers = pd.read_csv(c); orders = pd.read_csv(o); sessions = pd.read_csv(s)

orders["order_date"] = pd.to_datetime(orders["order_date"])
customers["signup_date"] = pd.to_datetime(customers["signup_date"])
sessions["session_date"] = pd.to_datetime(sessions["session_date"])

# Filters
min_d, max_d = orders["order_date"].min().date(), orders["order_date"].max().date()
window = st.sidebar.date_input("Order window", (min_d, max_d), min_value=min_d, max_value=max_d)
tiers = st.sidebar.multiselect("City tier", sorted(customers["city_tier"].unique()), default=sorted(customers["city_tier"].unique()))
chans = st.sidebar.multiselect("Acq channel", sorted(customers["acq_channel"].unique()), default=sorted(customers["acq_channel"].unique()))
customers = customers[customers["city_tier"].isin(tiers) & customers["acq_channel"].isin(chans)]
orders = orders[orders["order_date"].between(pd.to_datetime(window[0]), pd.to_datetime(window[1]))]
sessions = sessions[sessions["session_date"].between(pd.to_datetime(window[0]), pd.to_datetime(window[1]))]

# KPIs
active = orders["customer_id"].nunique()
total = customers["customer_id"].nunique()
gmv = orders["gmv"].sum()
avg_ord = orders.groupby("customer_id")["order_id"].count().mean() if active>0 else 0

ref_date = pd.to_datetime(window[1])
last = orders.sort_values("order_date").groupby("customer_id")["order_date"].last()
cust = customers.merge(last.rename("last_order_date"), on="customer_id", how="left")
cust["days_since_last"] = (ref_date - cust["last_order_date"]).dt.days
cust["days_since_last"] = cust["days_since_last"].fillna(9999)
cust["churn_label"] = (cust["days_since_last"] > 60).astype(int)
churn_rate = 100*cust["churn_label"].mean()

c1,c2,c3,c4 = st.columns(4)
c1.metric("Total Customers", f"{total:,}")
c2.metric("Active Customers", f"{active:,}")
c3.metric("GMV in Window (₹)", f"{gmv:,.0f}")
c4.metric("Churn Rate", f"{churn_rate:.1f}%")

st.divider()

# Cohort retention
st.subheader("Cohort Retention (signup month × months since first order)")
o = orders.merge(customers[["customer_id","signup_date"]], on="customer_id", how="inner")
o["cohort_month"] = o["signup_date"].dt.to_period("M").astype(str)
o["order_month"]  = o["order_date"].dt.to_period("M").astype(str)
first = o.groupby("customer_id")["order_month"].min().rename("first_order_month")
o = o.merge(first, on="customer_id", how="left")
o["cohort_idx"] = (pd.PeriodIndex(o["order_month"],freq="M") - pd.PeriodIndex(o["first_order_month"],freq="M")).astype(int)
ret = o.groupby(["cohort_month","cohort_idx"])["customer_id"].nunique().reset_index(name="active_users")
base = o.groupby("cohort_month")["customer_id"].nunique().reset_index(name="cohort_size")
ret = ret.merge(base, on="cohort_month"); ret["retention_rate"] = ret["active_users"]/ret["cohort_size"]
pivot = ret.pivot(index="cohort_month", columns="cohort_idx", values="retention_rate").fillna(0).round(3)
st.dataframe(pivot.style.background_gradient(cmap="Greens"), use_container_width=True)

st.divider()

# RFM + simple churn model
st.subheader("Churn Prediction (Logistic Regression)")
rfm = orders.groupby("customer_id").agg(recency=("order_date", lambda s: (ref_date - s.max()).days),
                                        frequency=("order_id","count"),
                                        monetary=("gmv","sum"),
                                        avg_discount=("discount_pct","mean")).reset_index()
# sessions last 30d
sessions["is30"] = (sessions["session_date"] >= (ref_date - pd.Timedelta(days=30))).astype(int)
sess = sessions.groupby("customer_id").agg(sessions_30=("is30","sum"),
                                           minutes_30=("minutes", "sum")).reset_index()

feat = cust[["customer_id","churn_label","city_tier","acq_channel","age"]].merge(rfm, on="customer_id", how="left").merge(sess, on="customer_id", how="left")
feat = feat.fillna({"recency":9999,"frequency":0,"monetary":0,"avg_discount":0,"sessions_30":0,"minutes_30":0})
feat = pd.get_dummies(feat, columns=["city_tier","acq_channel"], drop_first=True)

X = feat.drop(columns=["customer_id","churn_label"]); y = feat["churn_label"]
scaler = StandardScaler(); Xs = scaler.fit_transform(X.select_dtypes(include=[np.number]))
X_train,X_test,y_train,y_test = train_test_split(Xs,y,test_size=0.25,random_state=42, stratify=y)
model = LogisticRegression(max_iter=1000); model.fit(X_train,y_train)
proba = model.predict_proba(X_test)[:,1]; pred=(proba>=0.5).astype(int)
acc = accuracy_score(y_test,pred); auc = roc_auc_score(y_test,proba)
c1,c2 = st.columns(2); c1.metric("Accuracy", f"{acc*100:.1f}%"); c2.metric("ROC AUC", f"{auc:.3f}")

coef = pd.DataFrame({"feature":X.columns, "coef":model.coef_[0]}).sort_values("coef", key=lambda s: s.abs(), ascending=False).head(12)
st.write("Top features by absolute weight:"); st.dataframe(coef, use_container_width=True)

# score all
feat["churn_score"] = model.predict_proba(scaler.transform(X.select_dtypes(include=[np.number])))[:,1]
at_risk = feat[feat["churn_score"]>=0.7].sort_values("churn_score", ascending=False)[["customer_id","churn_score","recency","frequency","monetary","sessions_30"]]
st.subheader("At-risk customers (score ≥ 0.70)")
st.dataframe(at_risk.head(200), use_container_width=True)
st.download_button("Download churn scores", feat[["customer_id","churn_label","churn_score"]].to_csv(index=False).encode("utf-8"),
                   file_name="churn_scores.csv", mime="text/csv")

st.caption("Use scores to target re-engagement campaigns (e.g., coupons to high GMV but high-risk users).")
