-- =====================================================================
-- load_data.sql  -  load the CLEANED CSVs into the constrained schema
-- Run from the project root:   psql -d zomato -f sql/schema.sql
--                              psql -d zomato -f sql/load_data.sql
-- (schema.sql also inserts a few illustrative rows; TRUNCATE first so the real load is clean.)
-- Order matters (foreign keys).
-- =====================================================================
TRUNCATE customer_feedback, payments, order_items, orders, menu, promotions,
         delivery_partners, restaurants, customers, traffic, weather, cities CASCADE;

\copy cities             FROM 'data/cleaned/cities.csv'             DELIMITER ',' CSV HEADER
\copy customers          FROM 'data/cleaned/customers.csv'          DELIMITER ',' CSV HEADER
\copy restaurants        FROM 'data/cleaned/restaurants.csv'        DELIMITER ',' CSV HEADER
\copy menu               FROM 'data/cleaned/menu.csv'               DELIMITER ',' CSV HEADER
\copy delivery_partners  FROM 'data/cleaned/delivery_partners.csv'  DELIMITER ',' CSV HEADER
\copy promotions         FROM 'data/cleaned/promotions.csv'         DELIMITER ',' CSV HEADER
\copy orders             FROM 'data/cleaned/orders.csv'             DELIMITER ',' CSV HEADER
\copy order_items        FROM 'data/cleaned/order_items.csv'        DELIMITER ',' CSV HEADER
\copy payments           FROM 'data/cleaned/payments.csv'           DELIMITER ',' CSV HEADER
\copy customer_feedback  FROM 'data/cleaned/customer_feedback.csv'  DELIMITER ',' CSV HEADER
\copy weather            FROM 'data/cleaned/weather.csv'            DELIMITER ',' CSV HEADER
\copy traffic            FROM 'data/cleaned/traffic.csv'            DELIMITER ',' CSV HEADER

-- Integrity checks (all should return 0 rows / 0)
SELECT 'orders without customer'  AS check, COUNT(*) FROM orders o LEFT JOIN customers c  USING (CustomerID)  WHERE c.CustomerID  IS NULL
UNION ALL SELECT 'orders without restaurant', COUNT(*) FROM orders o LEFT JOIN restaurants r USING (RestaurantID) WHERE r.RestaurantID IS NULL
UNION ALL SELECT 'payments without order',    COUNT(*) FROM payments p LEFT JOIN orders o USING (OrderID) WHERE o.OrderID IS NULL
UNION ALL SELECT 'rows in orders (expect 20475)', COUNT(*) FROM orders;

-- Notes
--  * weather and traffic are NOT unique on (City, Date): dates were sampled with repeats in the source data,
--    so queries H1/H2 in business_queries.sql average per city-date before joining if you need one row per order.
--  * Query "last 6 months" (A1) uses CURRENT_DATE; the data ends 2024-12-31, so use a fixed date for meaningful output:
--    replace CURRENT_DATE with DATE '2024-12-31' in A1, E4, G3.
