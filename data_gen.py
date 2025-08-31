
from datetime import datetime
import numpy as np, pandas as pd
from pathlib import Path
np.random.seed(42)
OUT = Path(__file__).parent

def make_customers(n=8000, start="2024-01-01", end="2025-08-15"):
    s = pd.to_datetime(start); e = pd.to_datetime(end)
    signup = s + pd.to_timedelta(np.random.randint(0,(e-s).days+1,size=n), unit="D")
    return pd.DataFrame({
        "customer_id": np.arange(1,n+1),
        "signup_date": signup.date.astype(str),
        "city_tier": np.random.choice(["Tier-1","Tier-2","Tier-3"], n, p=[0.25,0.45,0.30]),
        "acq_channel": np.random.choice(["Organic","Ads","Referral","Social"], n, p=[0.45,0.30,0.15,0.10]),
        "gender": np.random.choice(["F","M"], n, p=[0.55,0.45]),
        "age": np.clip(np.random.normal(28,6.5,n).astype(int),18,55)
    })

def draw_value(): return round(np.random.gamma(3.0,400)+100,2)

def make_orders(customers, end="2025-08-15"):
    e = pd.to_datetime(end); rows=[]; oid=1
    for _,c in customers.iterrows():
        s = pd.to_datetime(c.signup_date); span=(e-s).days
        lam=0.6 + (0.4 if c.acq_channel=="Organic" else -0.1 if c.acq_channel=="Ads" else 0)
        lam += -0.05 if c.city_tier=="Tier-3" else 0
        n=int(np.clip(np.random.poisson(max(lam,0.1)*4),0,20))
        for _ in range(n):
            d = s + pd.to_timedelta(np.random.randint(0,max(span,1)), unit="D")
            disc=int(np.random.choice([0,5,10,15,20,25,30], p=[.2,.2,.2,.15,.15,.06,.04]))
            base=draw_value(); final=round(base*(1-disc/100),2)
            rows.append({"order_id":oid,"customer_id":int(c.customer_id),"order_date":d.date().astype(str),
                         "gmv":final,"discount_pct":disc,"payment_method":np.random.choice(["COD","Prepaid"],p=[.55,.45])})
            oid+=1
    return pd.DataFrame(rows)

def make_sessions(customers, end="2025-08-15"):
    e=pd.to_datetime(end); rows=[]
    for _,c in customers.iterrows():
        s=pd.to_datetime(c.signup_date); span=max(1,(e-s).days)
        rate={"Organic":2.2,"Ads":1.4,"Referral":1.8,"Social":1.6}[c.acq_channel]
        n=int(np.random.poisson(rate*span/7))
        for _ in range(n):
            d=s+pd.to_timedelta(np.random.randint(0,span), unit="D")
            rows.append({"customer_id":int(c.customer_id),"session_date":d.date().astype(str),
                         "minutes":float(np.clip(np.random.normal(6.5,3.0),1,25)),"channel":np.random.choice(["App","Web"],p=[.8,.2])})
    return pd.DataFrame(rows)

def label_churn(customers, orders, window_days=60):
    if orders.empty:
        customers["churn_label"]=1; return customers
    orders["order_date"]=pd.to_datetime(orders["order_date"]); ref=orders["order_date"].max()
    last=orders.sort_values("order_date").groupby("customer_id")["order_date"].last()
    customers=customers.merge(last.rename("last_order_date"),left_on="customer_id",right_index=True,how="left")
    customers["days_since_last_order"]=(ref-customers["last_order_date"]).dt.days
    customers["days_since_last_order"]=customers["days_since_last_order"].fillna(9999)
    customers["churn_label"]=(customers["days_since_last_order"]>window_days).astype(int)
    customers["reference_date"]=ref.date().astype(str)
    customers["last_order_date"]=customers["last_order_date"].dt.date.astype(str)
    return customers

def main():
    cust=make_customers(); ords=make_orders(cust); sess=make_sessions(cust); cust=label_churn(cust,ords)
    cust.to_csv(OUT/"customers.csv",index=False); ords.to_csv(OUT/"orders.csv",index=False); sess.to_csv(OUT/"sessions.csv",index=False)
    print("Data written to", OUT)

if __name__=="__main__": main()
