# Customer Retention & Churn Analysis
Deployable Streamlit project with synthetic data, cohort retention, and churn prediction.
Run:
```
pip install -r requirements.txt
streamlit run app.py
```

---
## Warehouse Options
This app can read from **Postgres** or **BigQuery** (besides CSVs).

### Postgres
Set `DATABASE_URL` (e.g., `postgresql+psycopg2://user:pass@host:5432/churndb`) in Streamlit **Secrets** or paste in the sidebar.

### BigQuery
Add these to Streamlit **Secrets**:
```toml
BQ_PROJECT = "your-project"
BQ_DATASET = "your_dataset"
GOOGLE_APPLICATION_CREDENTIALS = "gcp.json"
gcp.json = """
{ ...service account json... }
"""
```

## SQL
See `sql/cohort_queries.sql` for cohort retention calculation you can run in a warehouse.

## BI Templates (Superset & Metabase)

### Superset
- Connect your Postgres DB that has `customers`, `orders`, `sessions`.
- Import `superset/superset_dashboard.yaml` from **Settings → Import**.
- Update the database URI if needed.

### Metabase
- Import `metabase/metabase_collection.json` into a collection.
- Make sure the database points to the same Postgres with the 3 tables.

## Warehouse Bootstrap

### Postgres
1) Create tables and indexes:
```sql
\i sql/ddl_postgres.sql
```
2) Load CSVs (from your machine):
```bash
psql "$DATABASE_URL" -c "\copy customers(customer_id,signup_date,city_tier,acq_channel,gender,age) FROM 'customers.csv' CSV HEADER"
psql "$DATABASE_URL" -c "\copy orders(order_id,customer_id,order_date,gmv,discount_pct,payment_method) FROM 'orders.csv' CSV HEADER"
psql "$DATABASE_URL" -c "\copy sessions(customer_id,session_date,minutes,channel) FROM 'sessions.csv' CSV HEADER"
```
3) Backfill churn helpers:
```sql
\i sql/load_postgres.sql
```

### BigQuery
1) Create tables:
```sql
-- Replace {project}, {dataset} first
-- Then run the statements from sql/ddl_bigquery.sql
```
2) Load CSVs:
```bash
bq load --autodetect --replace --source_format=CSV <project>:<dataset>.customers customers.csv
bq load --autodetect --replace --source_format=CSV <project>:<dataset>.orders orders.csv
bq load --autodetect --replace --source_format=CSV <project>:<dataset>.sessions sessions.csv
```
3) Backfill churn helpers:
```sql
-- Use statements from sql/load_bigquery.sql and replace <project>, <dataset>
```

## BI Enhancements
- **Superset**: LTV metric added and an extra chart "LTV by Acquisition Channel".
- **Metabase**: new `dashboard_with_filters.json` with filters for **city_tier** and **acq_channel**.
