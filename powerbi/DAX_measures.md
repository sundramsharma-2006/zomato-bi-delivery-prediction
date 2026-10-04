# DAX measures (paste into a table named `_Measures`)

Revenue counts **completed** orders only (Delivered + Delivered Late). `fact_orders[IsDelivered]`, `[IsLate]`,
`[IsCancelled]`, `[HasCoupon]` are 0/1 flags created by the Python export, so every measure below is a plain SUM/COUNT.

```dax
-- ===== Core KPIs (Page 1) =====
Total Orders       = COUNTROWS ( fact_orders )
Completed Orders   = CALCULATE ( COUNTROWS ( fact_orders ), fact_orders[IsDelivered] = 1 )
Total Revenue      = CALCULATE ( SUM ( fact_orders[FinalAmount] ), fact_orders[IsDelivered] = 1 )
Avg Order Value    = DIVIDE ( [Total Revenue], [Completed Orders] )
Avg Delivery Time  = CALCULATE ( AVERAGE ( fact_orders[DeliveryTimeMinutes] ), fact_orders[IsDelivered] = 1 )
P90 Delivery Time  = PERCENTILEX.INC ( FILTER ( fact_orders, fact_orders[IsDelivered] = 1 && NOT ISBLANK ( fact_orders[DeliveryTimeMinutes] ) ),
                                       fact_orders[DeliveryTimeMinutes], 0.9 )
Avg Rating         = AVERAGE ( fact_orders[AvgRating] )
Late %             = DIVIDE ( CALCULATE ( COUNTROWS ( fact_orders ), fact_orders[IsLate] = 1 ), [Completed Orders] )
Cancellation %     = DIVIDE ( SUM ( fact_orders[IsCancelled] ), [Total Orders] )
Failure %          = DIVIDE ( CALCULATE ( COUNTROWS ( fact_orders ), NOT ( fact_orders[OrderStatus] IN { "Delivered", "Delivered Late" } ) ), [Total Orders] )

-- Active = placed an order in the 60 days up to the last date in the current filter context
Active Customers =
VAR EndD = MAX ( dim_date[Date] )
RETURN CALCULATE ( DISTINCTCOUNT ( fact_orders[CustomerID] ), DATESBETWEEN ( dim_date[Date], EndD - 59, EndD ) )

-- ===== Time intelligence =====
Revenue PM   = CALCULATE ( [Total Revenue], DATEADD ( dim_date[Date], -1, MONTH ) )
Revenue MoM% = DIVIDE ( [Total Revenue] - [Revenue PM], [Revenue PM] )
Revenue 30d Avg = AVERAGEX ( DATESINPERIOD ( dim_date[Date], MAX ( dim_date[Date] ), -30, DAY ), [Total Revenue] )

-- ===== Customer Analytics (Page 2) =====
Ordering Customers = CALCULATE ( COUNTROWS ( dim_customer ), dim_customer[Orders] >= 1 )
Repeat Customers   = CALCULATE ( COUNTROWS ( dim_customer ), dim_customer[Orders] >= 2 )
Retention Rate     = DIVIDE ( [Repeat Customers], [Ordering Customers] )
One-time Customers % = DIVIDE ( CALCULATE ( COUNTROWS ( dim_customer ), dim_customer[CustomerType] = "One-time" ), [Ordering Customers] )
Avg CLV            = CALCULATE ( AVERAGE ( dim_customer[CLV] ), dim_customer[Orders] >= 1 )
New Customers      = CALCULATE ( COUNTROWS ( dim_customer ), USERELATIONSHIP ( dim_customer[RegistrationDate], dim_date[Date] ) )  -- needs the inactive relationship below

-- ===== Restaurant Analytics (Page 3) =====
Avg Composite Score = AVERAGE ( dim_restaurant[CompositeScore] )
Restaurant Revenue  = [Total Revenue]
Restaurant Cancel % = [Cancellation %]
-- Bottom/Top-N: use a visual-level Top N filter on dim_restaurant[RestaurantName] by [Avg Composite Score];
-- add a visual filter  Orders >= 15  so rates are not driven by 5-order restaurants.

-- ===== Delivery Analytics (Page 4) =====
Partner Avg Time   = [Avg Delivery Time]
Partner Deliveries = [Completed Orders]
Traffic Impact     = [Avg Delivery Time]   -- put fact_orders[TrafficScore] on the axis
Rain Impact        = [Avg Delivery Time]   -- put fact_orders[RainImpact] on the axis

-- ===== Sales (Page 5) =====
Total Discount     = CALCULATE ( SUM ( fact_orders[Discount] ), fact_orders[IsDelivered] = 1 )
Coupon Usage %     = DIVIDE ( SUM ( fact_orders[HasCoupon] ), [Total Orders] )
Discount % of Revenue = DIVIDE ( [Total Discount], [Total Revenue] + [Total Discount] )

-- ===== ML (Page 6) =====
Delivery MAE   = AVERAGEX ( ml_delivery_predictions, ABS ( ml_delivery_predictions[Residual] ) )
Delivery RMSE  = SQRT ( AVERAGEX ( ml_delivery_predictions, ml_delivery_predictions[Residual] ^ 2 ) )
Delivery R2    = 1 - DIVIDE ( SUMX ( ml_delivery_predictions, ml_delivery_predictions[Residual] ^ 2 ),
                              SUMX ( ml_delivery_predictions, ( ml_delivery_predictions[Actual] - AVERAGE ( ml_delivery_predictions[Actual] ) ) ^ 2 ) )
High-risk Customers = CALCULATE ( COUNTROWS ( ml_churn_scores ), ml_churn_scores[RiskSegment] = "High", ml_churn_scores[Split] = "test" )
Actual Churn % (test) = CALCULATE ( AVERAGE ( ml_churn_scores[Churn] ), ml_churn_scores[Split] = "test" )
```

## Relationships (star schema, all single-direction, many-to-one toward the dimension)

| From (many) | To (one) | Notes |
|---|---|---|
| fact_orders[OrderDate] | dim_date[Date] | active; mark dim_date as the Date table |
| fact_orders[CustomerID] | dim_customer[CustomerID] | |
| fact_orders[RestaurantID] | dim_restaurant[RestaurantID] | |
| fact_orders[DeliveryPartnerID] | dim_partner[DeliveryPartnerID] | |
| fact_orders[City] | dim_city[City] | City = restaurant city; City slicer uses dim_city[City] |
| fact_orders[CouponCode] | dim_promotion[CouponCode] | blanks allowed (orders without a coupon) |
| fact_order_items[OrderID] | fact_orders[OrderID] | |
| fact_order_items[FoodItemID] | dim_menu[FoodItemID] | |
| dim_customer[RegistrationDate] | dim_date[Date] | **inactive** (used by `New Customers` via USERELATIONSHIP) |
| ml_delivery_predictions[OrderID] | fact_orders[OrderID] | |
| ml_churn_scores[CustomerID] | dim_customer[CustomerID] | |

`ml_*_feature_importance` and `ml_*_model_comparison` stay disconnected (used only as visual sources).
