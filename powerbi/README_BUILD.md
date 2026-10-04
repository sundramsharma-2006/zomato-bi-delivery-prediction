# Power BI build kit (6 pages)

## Quick start (Power BI Desktop installed)
1. Double-click **`powerbi/zomato_dashboard.pbip`** (or File > Open in Power BI Desktop).
   * If Desktop says the project format is a preview feature: File > Options and settings > Options > Preview features, tick
     **Power BI Project (.pbip) save options** and **Store semantic model using TMDL format**, restart, open again.
2. When prompted, click **Refresh** (data = the CSVs in `powerbi/data`). The `DataPath` parameter is preset to
   `C:\Users\hp\OneDrive\Desktop\internship\Zomato_Business_Intelligence_Project\powerbi\data`; if the project lives elsewhere:
   Home > Transform data > Manage parameters > `DataPath`.
3. Check the six pages, then **File > Save as > `zomato_dashboard.pbix`** (the deliverable the PRD asks for) in this `powerbi/` folder.

What is generated: 17 tables (CSV import, typed), 11 relationships (star schema, one inactive), 38 DAX measures (folder `KPIs`, `Customers`,
`Delivery`, `Sales`, `ML`...), the Zomato theme, 6 pages / 72 visuals (cards, line, bar, column, donut, scatter, table, City + Date slicers synced
across pages 1-5).

> **Honest status.** The project was generated from Microsoft's published PBIP/PBIR/TMDL schemas **without being opened in Power BI Desktop**
> (the build environment has no Desktop). JSON/TMDL syntax was validated and visual positions were checked for overlap, but not rendered.
> If Desktop reports an error, it names the file: fix or delete that visual folder under
> `zomato_dashboard.Report/definition/pages/<page>/visuals/`, or fall back to the manual build below (`DAX_measures.md` has every measure).
> Please verify numbers against the reports: Total Revenue INR 53.1 lakh, 20,475 orders, Late % 15.9%, Failure % 33.9%.

## Manual build (fallback)
Everything below is also described by the generated project; use it only if the PBIP does not open.

## 1. Load (NFR-04: refresh without re-mapping)
1. Home > Transform data > Manage parameters > New: `DataPath` (Text) = full path to `powerbi/data`.
2. For each CSV: Get data > Text/CSV, then in the Advanced Editor replace the path with `DataPath & "\\fact_orders.csv"`:
   ```
   let Source = Csv.Document(File.Contents(DataPath & "\fact_orders.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),
       Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),
       Typed = Table.TransformColumnTypes(Promoted, {{"OrderDate", type date}, {"FinalAmount", type number}, {"DeliveryTimeMinutes", type number}})
   in Typed
   ```
   Set date columns to **Date** (`OrderDate`, `RegistrationDate`, `FirstOrderDate`, `LastOrderDate`, `dim_date[Date]`). To refresh later: change `DataPath` only.
3. Tables: dim_date, dim_city, dim_customer, dim_restaurant, dim_partner, dim_promotion, dim_menu, fact_orders, fact_order_items, ml_* (7 files).
4. Model view: create the relationships in `DAX_measures.md`; mark `dim_date` as Date table. Create `_Measures` and paste the DAX.
5. View > Themes > Browse for themes > `zomato_theme.json`.

## 2. Pages (all pages: slicers **City** = dim_city[City] and **Date** = dim_date[Date] between-slicer, synced across pages)
| # | Page | Visuals |
|---|---|---|
| 1 | **Executive Dashboard** | 6 cards: Total Revenue, Total Orders, Avg Delivery Time, Avg Rating, Active Customers, Late %. Line: Total Revenue by dim_date[MonthStart]. Bar: Total Revenue by City (top 10). Donut: Orders by OrderStatus. Tooltip page: Failure %, Cancellation %, P90 Delivery Time. |
| 2 | **Customer Analytics** | Cards: Retention Rate, One-time Customers %, Avg CLV. Column: customers by `dim_customer[Segment]` (New / Active / At-Risk / Lapsed / No orders). Line: cumulative New Customers by quarter. Histogram (binned column): `dim_customer[CLV]`. |
| 3 | **Restaurant Analytics** | Two bar charts of `dim_restaurant[RestaurantName]` by Avg Composite Score: Top 10 and Bottom 10 (Top N filter), both with visual filter Orders >= 15. Bar: Total Revenue by `dim_restaurant[Cuisine]` with Avg Rating as a line. Table: city rank (`CityRank`, `CityRankFromBottom`). |
| 4 | **Delivery Analytics** | Table leaderboard: partner, Partner Deliveries, Partner Avg Time, Rating (filter deliveries >= 10). Line: Avg Delivery Time by month. Column: Avg Delivery Time by TrafficScore. Column: by RainImpact. |
| 5 | **Sales Dashboard** | Line: Total Revenue daily with Revenue 30d Avg; Bar: Total Discount by `dim_promotion[CampaignName]`; cards Coupon Usage %, Discount % of Revenue; Donut: orders by PaymentMethod. |
| 6 | **ML Dashboard** | Scatter: ml_delivery_predictions Predicted (x) vs Actual (y) + cards Delivery R2 / RMSE / MAE; Bar: `ml_delivery_feature_importance` (permutation_mse_increase); Column: customers by `ml_churn_scores[RiskSegment]` (Split = test) and Actual Churn % per segment; Bar: `ml_churn_feature_importance`. **Add a text box on this page stating the model results (R2 ~ 0.00, ROC-AUC ~ 0.53)** - see reports/business_insights_report.pdf. |

## 3. Honest-reporting notes for the dashboard
* Page 4 and 6 will show flat bars/scatter: that *is* the finding. Do not add trend lines or colour scales that exaggerate noise.
* Page 3: scorecard rates are noisy (median restaurant has 17 orders); keep the Orders >= 15 filter visible.
* Weather/traffic cover ~63% of orders (`HasWeather`/`HasTraffic` flags in the Python ABT, not exported to the fact table).
