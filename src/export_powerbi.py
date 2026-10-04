"""export_powerbi.py - Star-schema CSVs for Power BI (NFR-04: refresh from cleaned CSVs, no re-mapping).

Fact: fact_orders (+ fact_order_items)   Dimensions: dim_date, dim_city, dim_customer,
dim_restaurant, dim_partner, dim_promotion   ML tables: ml_* (model outputs).
Run: python -m src.export_powerbi
"""
import numpy as np
import pandas as pd
from .ingest import load_cleaned, ROOT
from .features import build_order_base

OUT = ROOT / "powerbi" / "data"
AS_OF = None


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    c = load_cleaned(); abt = build_order_base(c); cl = ROOT / "data" / "cleaned"
    as_of = abt.OrderDate.max()

    d = pd.date_range("2023-01-01", "2024-12-31")
    dim_date = pd.DataFrame({"Date": d, "Year": d.year, "Quarter": "Q" + d.quarter.astype(str), "MonthNum": d.month,
                             "MonthName": d.strftime("%b"), "YearMonth": d.strftime("%Y-%m"), "MonthStart": d.to_period("M").to_timestamp(),
                             "DayOfWeek": d.strftime("%a"), "DayOfWeekNum": d.dayofweek + 1, "IsWeekend": (d.dayofweek >= 5).astype(int)})
    dim_date.to_csv(OUT / "dim_date.csv", index=False)
    c["cities"].to_csv(OUT / "dim_city.csv", index=False)

    g = abt.groupby("CustomerID")
    cu = c["customers"].rename(columns={"City": "CustomerCity"}).merge(pd.DataFrame({
        "Orders": g.OrderID.count(), "CompletedOrders": g.IsDelivered.sum(), "CLV": g.apply(lambda x: x.loc[x.IsDelivered, "FinalAmount"].sum(), include_groups=False),
        "FirstOrderDate": g.OrderDate.min(), "LastOrderDate": g.OrderDate.max()}).reset_index(), on="CustomerID", how="left")
    for col in ["Phone", "Pincode"]:                                   # identifiers, not numbers: keep as clean text
        cu[col] = pd.to_numeric(cu[col], errors="coerce").astype("Int64").astype("string")
    cu["Orders"] = cu.Orders.fillna(0).astype(int); cu["CLV"] = cu.CLV.fillna(0)
    cu["DaysSinceLastOrder"] = (as_of - cu.LastOrderDate).dt.days
    cu["Segment"] = np.select([cu.Orders == 0, (as_of - cu.FirstOrderDate).dt.days <= 60, cu.DaysSinceLastOrder <= 60, cu.DaysSinceLastOrder <= 180],
                              ["No orders", "New", "Active", "At-Risk"], default="Lapsed")
    cu["CustomerType"] = np.where(cu.Orders >= 2, "Repeat", np.where(cu.Orders == 1, "One-time", "No orders"))
    bins = [-0.01, 250, 500, 750, 1000, 1500, 2000, 1e9]; labels = ["0-250", "250-500", "500-750", "750-1000", "1000-1500", "1500-2000", "2000+"]
    cu["CLVBand"] = pd.cut(cu.CLV, bins=bins, labels=labels).astype(str); cu["CLVBandOrder"] = pd.cut(cu.CLV, bins=bins, labels=False).astype(int) + 1
    cu.to_csv(OUT / "dim_customer.csv", index=False)

    sc = pd.read_csv(cl / "restaurant_scorecard.csv")[["RestaurantID", "CompositeScore", "CityRank", "CityRankFromBottom"]]
    c["restaurants"].merge(sc, on="RestaurantID", how="left").to_csv(OUT / "dim_restaurant.csv", index=False)
    full = pd.read_csv(cl / "restaurant_scorecard.csv"); full = full[full.Orders >= 15].copy()          # >=15 orders so rates are not pure noise
    full["RestaurantLabel"] = full.RestaurantName + " (" + full.RCity + ")"
    cols = ["RestaurantID", "RestaurantLabel", "RCity", "Cuisine", "Orders", "Rating", "CancelRate", "LateRate", "CompositeScore"]
    full.nlargest(10, "CompositeScore")[cols].to_csv(OUT / "rpt_restaurant_top10.csv", index=False)
    full.nsmallest(10, "CompositeScore")[cols].to_csv(OUT / "rpt_restaurant_bottom10.csv", index=False)
    c["delivery_partners"].to_csv(OUT / "dim_partner.csv", index=False)
    c["promotions"].to_csv(OUT / "dim_promotion.csv", index=False)

    f = abt[["OrderID", "OrderDate", "OrderTime", "hour", "CustomerID", "RestaurantID", "DeliveryPartnerID", "RCity", "CouponCode", "OrderStatus",
             "PaymentMethod", "PaymentStatus", "FoodCost", "DeliveryFee", "Discount", "GST", "FinalAmount", "DeliveryTimeMinutes", "BasketSize",
             "IsDelivered", "IsCancelled", "IsLate", "HasCoupon", "PeakHour", "WeekendOrder", "TrafficScore", "RainImpact", "Rainfall", "Temperature",
             "WeatherCondition", "DeliveryEfficiency", "AvgRating", "CustomerRating", "DeliveryRating", "FoodRating", "Sentiment"]].rename(columns={"RCity": "City"})
    f["IsDelivered"] = f.IsDelivered.astype(int); f["Revenue"] = np.where(f.IsDelivered == 1, f.FinalAmount, 0.0)
    f.to_csv(OUT / "fact_orders.csv", index=False)
    c["order_items"].to_csv(OUT / "fact_order_items.csv", index=False); c["menu"].to_csv(OUT / "dim_menu.csv", index=False)

    for src, dst in [("delivery_predictions_test", "ml_delivery_predictions"), ("feature_importance_delivery", "ml_delivery_feature_importance"),
                     ("model_comparison_delivery", "ml_delivery_model_comparison"), ("churn_scores", "ml_churn_scores"),
                     ("feature_importance_churn", "ml_churn_feature_importance"), ("model_comparison_churn", "ml_churn_model_comparison")]:
        p = cl / f"{src}.csv"
        if p.exists():
            t = pd.read_csv(p)
            if "feature_importance" in dst: t = t.head(12)          # dashboard shows the 12 largest only
            t.to_csv(OUT / f"{dst}.csv", index=False)
    print("exported:", sorted(p.name for p in OUT.glob("*.csv")))


if __name__ == "__main__":
    main()
