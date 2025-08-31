
-- Postgres DDL for churn project
CREATE TABLE IF NOT EXISTS customers (
  customer_id    BIGINT PRIMARY KEY,
  signup_date    DATE NOT NULL,
  city_tier      TEXT,
  acq_channel    TEXT,
  gender         TEXT,
  age            INTEGER,
  last_order_date DATE,
  days_since_last_order INTEGER,
  churn_label    INTEGER,
  reference_date DATE
);

CREATE TABLE IF NOT EXISTS orders (
  order_id       BIGINT PRIMARY KEY,
  customer_id    BIGINT REFERENCES customers(customer_id),
  order_date     DATE NOT NULL,
  gmv            NUMERIC(12,2) NOT NULL,
  discount_pct   INTEGER,
  payment_method TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
  customer_id    BIGINT REFERENCES customers(customer_id),
  session_date   DATE NOT NULL,
  minutes        NUMERIC(8,2),
  channel        TEXT
);

-- Helpful indexes
CREATE INDEX IF NOT EXISTS idx_orders_date ON orders(order_date);
CREATE INDEX IF NOT EXISTS idx_orders_customer ON orders(customer_id);
CREATE INDEX IF NOT EXISTS idx_sessions_customer ON sessions(customer_id);
