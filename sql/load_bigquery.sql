
-- BigQuery loading helpers (from local files via bq CLI)
-- Replace <project>, <dataset> and paths.
--
-- bq load --autodetect --replace --source_format=CSV <project>:<dataset>.customers ./customers.csv
-- bq load --autodetect --replace --source_format=CSV <project>:<dataset>.orders ./orders.csv
-- bq load --autodetect --replace --source_format=CSV <project>:<dataset>.sessions ./sessions.csv
--
-- Backfill churn helpers:
CREATE OR REPLACE TABLE `<project>.<dataset>.customers` AS
SELECT c.* EXCEPT(last_order_date, days_since_last_order, churn_label, reference_date),
       last_order_date,
       IFNULL(DATE_DIFF(reference_date, last_order_date, DAY), 9999) AS days_since_last_order,
       IF(IFNULL(DATE_DIFF(reference_date, last_order_date, DAY), 9999) > 60, 1, 0) AS churn_label,
       reference_date
FROM (
  SELECT c.*,
         (SELECT MAX(order_date) FROM `<project>.<dataset>.orders` o WHERE o.customer_id = c.customer_id) AS last_order_date,
         DATE(MAX(order_date)) OVER () AS reference_date
  FROM `<project>.<dataset>.customers` c
);
