
-- Postgres loading helpers
-- Option A: psql from your machine (CSV paths are local to your psql client)
-- Ensure CSVs have headers.
-- \copy is a psql command (not standard SQL). Run from terminal:
--
--   psql "$DATABASE_URL" -c "\copy customers(customer_id,signup_date,city_tier,acq_channel,gender,age) FROM 'customers.csv' CSV HEADER"
--   psql "$DATABASE_URL" -c "\copy orders(order_id,customer_id,order_date,gmv,discount_pct,payment_method) FROM 'orders.csv' CSV HEADER"
--   psql "$DATABASE_URL" -c "\copy sessions(customer_id,session_date,minutes,channel) FROM 'sessions.csv' CSV HEADER"
--
-- Option B: if files are accessible to server (rare in managed DBs), use COPY:
-- COPY customers(customer_id,signup_date,city_tier,acq_channel,gender,age) FROM '/abs/path/customers.csv' CSV HEADER;
-- COPY orders(order_id,customer_id,order_date,gmv,discount_pct,payment_method) FROM '/abs/path/orders.csv' CSV HEADER;
-- COPY sessions(customer_id,session_date,minutes,channel) FROM '/abs/path/sessions.csv' CSV HEADER;
--
-- After loading orders, you can backfill churn helper columns:
UPDATE customers c
SET last_order_date = o.last_dt
FROM (
  SELECT customer_id, MAX(order_date) AS last_dt
  FROM orders GROUP BY 1
) o
WHERE c.customer_id = o.customer_id;

UPDATE customers
SET days_since_last_order = (reference_date - last_order_date),
    churn_label = CASE WHEN (reference_date - last_order_date) > 60 THEN 1 ELSE 0 END
WHERE last_order_date IS NOT NULL;
