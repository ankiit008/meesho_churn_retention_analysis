
-- BigQuery DDL for churn project
CREATE SCHEMA IF NOT EXISTS `{project}.{dataset}`;

CREATE OR REPLACE TABLE `{project}.{dataset}.customers` (
  customer_id INT64,
  signup_date DATE,
  city_tier STRING,
  acq_channel STRING,
  gender STRING,
  age INT64,
  last_order_date DATE,
  days_since_last_order INT64,
  churn_label INT64,
  reference_date DATE
);

CREATE OR REPLACE TABLE `{project}.{dataset}.orders` (
  order_id INT64,
  customer_id INT64,
  order_date DATE,
  gmv NUMERIC,
  discount_pct INT64,
  payment_method STRING
);

CREATE OR REPLACE TABLE `{project}.{dataset}.sessions` (
  customer_id INT64,
  session_date DATE,
  minutes NUMERIC,
  channel STRING
);
