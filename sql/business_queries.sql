-- =====================================================================
-- ZOMATO BUSINESS INTELLIGENCE & DELIVERY TIME PREDICTION PLATFORM
-- Business Query Bank (run against the cleaned, constrained schema
-- created in schema.sql)
-- Target engine: PostgreSQL 14+  (MySQL-compatible notes inline)
-- =====================================================================


-- =====================================================================
-- SECTION A: CORE QUERYING -- SELECT, WHERE, GROUP BY, ORDER BY, HAVING
-- =====================================================================

-- A1. Top 10 restaurants by revenue in the last 6 months
SELECT
    r.RestaurantID,
    r.RestaurantName,
    r.City,
    SUM(o.FinalAmount) AS total_revenue,
    COUNT(o.OrderID) AS total_orders
FROM orders o
JOIN restaurants r ON r.RestaurantID = o.RestaurantID
WHERE o.OrderDate >= (DATE '2024-12-31' - INTERVAL '6 months')
  AND o.OrderStatus = 'Delivered'
GROUP BY r.RestaurantID, r.RestaurantName, r.City
ORDER BY total_revenue DESC
LIMIT 10;

-- A2. Average delivery time by city, slowest first
SELECT
    r.City,
    ROUND(AVG(o.DeliveryTimeMinutes), 2) AS avg_delivery_time,
    COUNT(*) AS order_count
FROM orders o
JOIN restaurants r ON r.RestaurantID = o.RestaurantID
WHERE o.DeliveryTimeMinutes IS NOT NULL
GROUP BY r.City
ORDER BY avg_delivery_time DESC;

-- A3. Cuisines with more than 500 delivered orders and average rating above 4.0
SELECT
    r.Cuisine,
    COUNT(o.OrderID) AS delivered_orders,
    ROUND(AVG(r.Rating), 2) AS avg_cuisine_rating
FROM orders o
JOIN restaurants r ON r.RestaurantID = o.RestaurantID
WHERE o.OrderStatus = 'Delivered'
GROUP BY r.Cuisine
HAVING COUNT(o.OrderID) > 500 AND AVG(r.Rating) > 4.0
ORDER BY delivered_orders DESC;

-- A4. Cancellation rate by restaurant cuisine
SELECT
    r.Cuisine,
    COUNT(*) FILTER (WHERE o.OrderStatus = 'Cancelled') AS cancelled_orders,   -- MySQL: use SUM(CASE WHEN ... THEN 1 ELSE 0 END)
    COUNT(*) AS total_orders,
    ROUND(100.0 * COUNT(*) FILTER (WHERE o.OrderStatus = 'Cancelled') / COUNT(*), 2) AS cancellation_rate_pct
FROM orders o
JOIN restaurants r ON r.RestaurantID = o.RestaurantID
GROUP BY r.Cuisine
ORDER BY cancellation_rate_pct DESC;


-- =====================================================================
-- SECTION B: JOINS -- INNER, LEFT, RIGHT, SELF, UNION
-- =====================================================================

-- B1. INNER JOIN: orders with customer and restaurant names
SELECT
    o.OrderID, c.Name AS customer_name, r.RestaurantName, o.FinalAmount, o.OrderStatus
FROM orders o
INNER JOIN customers c   ON c.CustomerID   = o.CustomerID
INNER JOIN restaurants r ON r.RestaurantID = o.RestaurantID
LIMIT 100;

-- B2. LEFT JOIN: every restaurant with its feedback count (including restaurants with zero feedback)
SELECT
    r.RestaurantID,
    r.RestaurantName,
    COUNT(f.FeedbackID) AS feedback_count,
    ROUND(AVG(f.CustomerRating), 2) AS avg_customer_rating
FROM restaurants r
LEFT JOIN orders o            ON o.RestaurantID = r.RestaurantID
LEFT JOIN customer_feedback f ON f.OrderID = o.OrderID
GROUP BY r.RestaurantID, r.RestaurantName
ORDER BY feedback_count ASC;

-- B3. RIGHT JOIN: every payment with its order context (including any orphaned payment rows)
-- (Written as a LEFT JOIN with table order swapped, which is the portable equivalent
--  since MySQL/Postgres both support RIGHT JOIN but LEFT JOIN is more commonly indexed)
SELECT
    p.PaymentID, p.OrderID, p.PaymentStatus, o.FinalAmount, o.OrderStatus
FROM orders o
RIGHT JOIN payments p ON p.OrderID = o.OrderID
ORDER BY p.PaymentID
LIMIT 100;

-- B4. SELF JOIN: delivery partners in the same city paired together (for workload comparison)
SELECT
    dp1.DeliveryPartnerID AS partner_a,
    dp2.DeliveryPartnerID AS partner_b,
    dp1.City,
    dp1.AverageDeliveryTime AS a_avg_time,
    dp2.AverageDeliveryTime AS b_avg_time
FROM delivery_partners dp1
JOIN delivery_partners dp2
  ON dp1.City = dp2.City
 AND dp1.DeliveryPartnerID < dp2.DeliveryPartnerID
WHERE ABS(dp1.AverageDeliveryTime - dp2.AverageDeliveryTime) > 20
LIMIT 100;

-- B5. UNION: a combined "attention list" of low-rated restaurants and low-rated delivery partners
SELECT RestaurantID AS entity_id, RestaurantName AS entity_name, 'Restaurant' AS entity_type, Rating
FROM restaurants
WHERE Rating < 3.0
UNION
SELECT DeliveryPartnerID AS entity_id, Name AS entity_name, 'DeliveryPartner' AS entity_type, Rating
FROM delivery_partners
WHERE Rating < 3.0
ORDER BY Rating ASC;


-- =====================================================================
-- SECTION C: CASE EXPRESSIONS
-- =====================================================================

-- C1. Bucket orders into delivery-speed tiers
SELECT
    OrderID,
    DeliveryTimeMinutes,
    CASE
        WHEN DeliveryTimeMinutes IS NULL THEN 'Unknown'
        WHEN DeliveryTimeMinutes <= 20 THEN 'Fast'
        WHEN DeliveryTimeMinutes <= 40 THEN 'On Time'
        WHEN DeliveryTimeMinutes <= 60 THEN 'Delayed'
        ELSE 'Severely Delayed'
    END AS delivery_tier
FROM orders;

-- C2. Customer value segments based on lifetime spend
SELECT
    c.CustomerID,
    c.Name,
    SUM(o.FinalAmount) AS lifetime_spend,
    CASE
        WHEN SUM(o.FinalAmount) >= 20000 THEN 'Platinum'
        WHEN SUM(o.FinalAmount) >= 10000 THEN 'Gold'
        WHEN SUM(o.FinalAmount) >= 3000  THEN 'Silver'
        ELSE 'Bronze'
    END AS value_segment
FROM customers c
JOIN orders o ON o.CustomerID = c.CustomerID
GROUP BY c.CustomerID, c.Name
ORDER BY lifetime_spend DESC;


-- =====================================================================
-- SECTION D: SUBQUERIES AND CTEs
-- =====================================================================

-- D1. Scalar subquery: restaurants priced above the platform-wide average cost
SELECT RestaurantID, RestaurantName, AverageCost
FROM restaurants
WHERE AverageCost > (SELECT AVG(AverageCost) FROM restaurants)
ORDER BY AverageCost DESC;

-- D2. Correlated subquery: customers whose most recent order was cancelled
SELECT c.CustomerID, c.Name
FROM customers c
WHERE (
    SELECT o.OrderStatus
    FROM orders o
    WHERE o.CustomerID = c.CustomerID
    ORDER BY o.OrderDate DESC, o.OrderTime DESC
    LIMIT 1
) = 'Cancelled';

-- D3. CTE: monthly revenue trend per city
WITH monthly_revenue AS (
    SELECT
        r.City,
        DATE_TRUNC('month', o.OrderDate) AS order_month,
        SUM(o.FinalAmount) AS revenue
    FROM orders o
    JOIN restaurants r ON r.RestaurantID = o.RestaurantID
    WHERE o.OrderStatus = 'Delivered'
    GROUP BY r.City, DATE_TRUNC('month', o.OrderDate)
)
SELECT City, order_month, revenue,
       LAG(revenue) OVER (PARTITION BY City ORDER BY order_month) AS prev_month_revenue
FROM monthly_revenue
ORDER BY City, order_month;

-- D4. Multi-CTE: repeat-customer identification feeding into a segment summary
WITH order_counts AS (
    SELECT CustomerID, COUNT(*) AS order_count
    FROM orders
    GROUP BY CustomerID
),
customer_segments AS (
    SELECT
        CustomerID,
        CASE WHEN order_count = 1 THEN 'One-Time' ELSE 'Repeat' END AS segment
    FROM order_counts
)
SELECT segment, COUNT(*) AS num_customers
FROM customer_segments
GROUP BY segment;


-- =====================================================================
-- SECTION E: WINDOW FUNCTIONS -- ROW_NUMBER, RANK, DENSE_RANK, LAG, LEAD
-- =====================================================================

-- E1. Rank restaurants within each city by revenue
SELECT
    RestaurantID, RestaurantName, City, total_revenue,
    ROW_NUMBER() OVER (PARTITION BY City ORDER BY total_revenue DESC) AS row_num,
    RANK()       OVER (PARTITION BY City ORDER BY total_revenue DESC) AS rank_num,
    DENSE_RANK() OVER (PARTITION BY City ORDER BY total_revenue DESC) AS dense_rank_num
FROM (
    SELECT r.RestaurantID, r.RestaurantName, r.City, SUM(o.FinalAmount) AS total_revenue
    FROM restaurants r
    JOIN orders o ON o.RestaurantID = r.RestaurantID
    WHERE o.OrderStatus = 'Delivered'
    GROUP BY r.RestaurantID, r.RestaurantName, r.City
) ranked;

-- E2. Top 3 restaurants per city (using the ranking above as a filter)
WITH city_ranks AS (
    SELECT
        r.RestaurantID, r.RestaurantName, r.City,
        SUM(o.FinalAmount) AS total_revenue,
        RANK() OVER (PARTITION BY r.City ORDER BY SUM(o.FinalAmount) DESC) AS city_rank
    FROM restaurants r
    JOIN orders o ON o.RestaurantID = r.RestaurantID
    WHERE o.OrderStatus = 'Delivered'
    GROUP BY r.RestaurantID, r.RestaurantName, r.City
)
SELECT * FROM city_ranks WHERE city_rank <= 3
ORDER BY City, city_rank;

-- E3. LAG/LEAD: month-over-month change in a customer's order value
SELECT
    CustomerID,
    OrderDate,
    FinalAmount,
    LAG(FinalAmount)  OVER (PARTITION BY CustomerID ORDER BY OrderDate) AS previous_order_amount,
    LEAD(FinalAmount) OVER (PARTITION BY CustomerID ORDER BY OrderDate) AS next_order_amount,
    FinalAmount - LAG(FinalAmount) OVER (PARTITION BY CustomerID ORDER BY OrderDate) AS change_vs_previous
FROM orders
WHERE OrderStatus = 'Delivered';

-- E4. Recency ranking: days since each customer's most recent order
SELECT
    CustomerID,
    MAX(OrderDate) AS last_order_date,
    DATE '2024-12-31' - MAX(OrderDate) AS days_since_last_order,
    RANK() OVER (ORDER BY MAX(OrderDate) ASC) AS recency_rank
FROM orders
GROUP BY CustomerID
ORDER BY days_since_last_order DESC;


-- =====================================================================
-- SECTION F: VIEWS
-- =====================================================================

-- F1. View: order-level analytical base table (joins the core tables once)
CREATE OR REPLACE VIEW vw_order_base AS
SELECT
    o.OrderID, o.CustomerID, o.RestaurantID, o.DeliveryPartnerID,
    o.OrderDate, o.OrderTime, o.DeliveryTimeMinutes, o.FinalAmount, o.OrderStatus,
    c.City AS customer_city, r.City AS restaurant_city, r.Cuisine, r.Rating AS restaurant_rating,
    dp.VehicleType, dp.Rating AS partner_rating
FROM orders o
JOIN customers c          ON c.CustomerID = o.CustomerID
JOIN restaurants r        ON r.RestaurantID = o.RestaurantID
JOIN delivery_partners dp ON dp.DeliveryPartnerID = o.DeliveryPartnerID;

-- Example usage:
-- SELECT restaurant_city, AVG(DeliveryTimeMinutes) FROM vw_order_base GROUP BY restaurant_city;

-- F2. View: restaurant performance scorecard
CREATE OR REPLACE VIEW vw_restaurant_scorecard AS
SELECT
    r.RestaurantID,
    r.RestaurantName,
    r.City,
    r.Rating,
    COUNT(o.OrderID) AS total_orders,
    COUNT(*) FILTER (WHERE o.OrderStatus = 'Cancelled') AS cancelled_orders,
    ROUND(100.0 * COUNT(*) FILTER (WHERE o.OrderStatus = 'Cancelled') / NULLIF(COUNT(o.OrderID), 0), 2) AS cancellation_rate_pct,
    ROUND(AVG(o.DeliveryTimeMinutes), 2) AS avg_delivery_time
FROM restaurants r
LEFT JOIN orders o ON o.RestaurantID = r.RestaurantID
GROUP BY r.RestaurantID, r.RestaurantName, r.City, r.Rating;


-- =====================================================================
-- SECTION G: DATE FUNCTIONS
-- =====================================================================

-- G1. Orders by day-of-week to check weekend vs weekday patterns
-- MySQL 8+ equivalent: replace TO_CHAR(OrderDate, 'Day') with DATE_FORMAT(OrderDate, '%W')
--                      replace EXTRACT(ISODOW FROM OrderDate) with WEEKDAY(OrderDate)
SELECT
    TO_CHAR(OrderDate, 'Day') AS day_of_week,
    COUNT(*) AS order_count,
    SUM(FinalAmount) AS total_revenue
FROM orders
GROUP BY TO_CHAR(OrderDate, 'Day'), EXTRACT(ISODOW FROM OrderDate)
ORDER BY EXTRACT(ISODOW FROM OrderDate);

-- G2. Quarter-level revenue rollup
SELECT
    EXTRACT(YEAR FROM OrderDate)    AS order_year,
    EXTRACT(QUARTER FROM OrderDate) AS order_quarter,
    SUM(FinalAmount) AS revenue
FROM orders
WHERE OrderStatus = 'Delivered'
GROUP BY 1, 2
ORDER BY 1, 2;

-- G3. Customer tenure in days as of today
SELECT
    CustomerID,
    RegistrationDate,
    DATE '2024-12-31' - RegistrationDate AS tenure_days
FROM customers
ORDER BY tenure_days DESC;


-- =====================================================================
-- SECTION H: AGGREGATIONS ACROSS MULTIPLE TABLES (WEATHER / TRAFFIC IMPACT)
-- =====================================================================

-- H1. Average delivery time by weather condition
SELECT
    w.WeatherCondition,
    ROUND(AVG(o.DeliveryTimeMinutes), 2) AS avg_delivery_time,
    COUNT(*) AS order_count
FROM orders o
JOIN restaurants r ON r.RestaurantID = o.RestaurantID
JOIN weather w      ON w.City = r.City AND w.Date = o.OrderDate
GROUP BY w.WeatherCondition
ORDER BY avg_delivery_time DESC;

-- H2. Average delivery time by traffic level
SELECT
    t.TrafficLevel,
    ROUND(AVG(o.DeliveryTimeMinutes), 2) AS avg_delivery_time,
    COUNT(*) AS order_count
FROM orders o
JOIN restaurants r ON r.RestaurantID = o.RestaurantID
JOIN traffic t      ON t.City = r.City AND t.Date = o.OrderDate
GROUP BY t.TrafficLevel
ORDER BY avg_delivery_time DESC;

-- H3. Coupon performance: total discount value and order volume per campaign
SELECT
    p.CampaignName,
    p.CouponCode,
    COUNT(o.OrderID) AS orders_using_coupon,
    SUM(o.Discount) AS total_discount_value,
    SUM(o.FinalAmount) AS revenue_from_coupon_orders
FROM promotions p
JOIN orders o ON o.CouponCode = p.CouponCode
GROUP BY p.CampaignName, p.CouponCode
ORDER BY total_discount_value DESC;

-- H4. Delivery partner leaderboard: top 10% by completed deliveries and rating
SELECT *
FROM (
    SELECT
        DeliveryPartnerID, Name, City, Rating, CompletedDeliveries,
        NTILE(10) OVER (ORDER BY CompletedDeliveries DESC, Rating DESC) AS decile
    FROM delivery_partners
) ranked
WHERE decile = 1
ORDER BY CompletedDeliveries DESC;


-- =====================================================================
-- SECTION I: OPTIONAL / STRETCH -- STORED PROCEDURE AND TRIGGER
-- =====================================================================

-- I1. Stored procedure (PostgreSQL): refresh a summary table of city-level KPIs
CREATE TABLE IF NOT EXISTS city_kpi_summary (
    City VARCHAR(100) PRIMARY KEY,
    TotalOrders INTEGER,
    TotalRevenue NUMERIC(14,2),
    AvgDeliveryTime NUMERIC(6,2),
    LastRefreshed TIMESTAMP
);

CREATE OR REPLACE PROCEDURE refresh_city_kpi_summary()
LANGUAGE plpgsql
AS $$
BEGIN
    DELETE FROM city_kpi_summary;

    INSERT INTO city_kpi_summary (City, TotalOrders, TotalRevenue, AvgDeliveryTime, LastRefreshed)
    SELECT
        r.City,
        COUNT(o.OrderID),
        SUM(o.FinalAmount),
        AVG(o.DeliveryTimeMinutes),
        NOW()
    FROM restaurants r
    JOIN orders o ON o.RestaurantID = r.RestaurantID
    GROUP BY r.City;
END;
$$;

-- Usage: CALL refresh_city_kpi_summary();

-- I2. Trigger (PostgreSQL): keep customers.TotalOrders in sync when a new order is inserted
CREATE OR REPLACE FUNCTION trg_increment_customer_orders()
RETURNS TRIGGER
LANGUAGE plpgsql
AS $$
BEGIN
    UPDATE customers
    SET TotalOrders = TotalOrders + 1
    WHERE CustomerID = NEW.CustomerID;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS trg_after_order_insert ON orders;
CREATE TRIGGER trg_after_order_insert
AFTER INSERT ON orders
FOR EACH ROW
EXECUTE FUNCTION trg_increment_customer_orders();

-- MySQL 8+ equivalents:
--   * Stored procedures use: DELIMITER $$ ... CREATE PROCEDURE name() BEGIN ... END $$ DELIMITER ;
--   * Triggers use: CREATE TRIGGER trg_name AFTER INSERT ON orders FOR EACH ROW BEGIN ... END;
--   * MySQL has no CREATE OR REPLACE PROCEDURE -- DROP PROCEDURE IF EXISTS first.
