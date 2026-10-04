"""features.py - Build the order-level analytical base table (ABT) and
customer-level churn table.

Order-level features (PRD section 23):
  PeakHour, WeekendOrder, RainImpact, TrafficScore, CustomerTenure,
  AverageBasketValue, RestaurantPopularity, DeliveryEfficiency,
  CustomerLifetimeValue, OrderFrequency, AverageRating (+ BasketSize, DistanceProxy)

LEAKAGE RULES
  * OrderStatus / feedback ratings are only known AFTER delivery, so they are
    never used as predictors of DeliveryTimeMinutes (see model_delivery.py).
  * Churn features are computed only from orders strictly before the cutoff date;
    the label is built from the 60 days after it.
"""
import numpy as np
import pandas as pd

TRAFFIC_SCORE = {"Low": 1, "Moderate": 2, "High": 3, "Severe": 4}
COMPLETED = ["Delivered", "Delivered Late"]


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorised great-circle distance in km."""
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def _attach_traffic(d: pd.DataFrame, traffic: pd.DataFrame) -> pd.DataFrame:
    """Join traffic by (restaurant city, date) and keep the reading closest in time of day."""
    t = traffic.copy()
    t["t_min"] = pd.to_datetime(t["Time"], format="%H:%M", errors="coerce").dt.hour * 60 + \
        pd.to_datetime(t["Time"], format="%H:%M", errors="coerce").dt.minute
    t["TrafficScore"] = t["TrafficLevel"].map(TRAFFIC_SCORE)
    m = d[["OrderID", "RCity", "OrderDate", "order_min"]].merge(
        t[["City", "Date", "t_min", "TrafficScore", "AverageSpeed"]],
        left_on=["RCity", "OrderDate"], right_on=["City", "Date"], how="inner")
    m["gap"] = (m["order_min"] - m["t_min"]).abs()
    m = m.sort_values(["OrderID", "gap"]).drop_duplicates("OrderID")
    return d.merge(m[["OrderID", "TrafficScore", "AverageSpeed"]], on="OrderID", how="left")


def build_order_base(c: dict) -> pd.DataFrame:
    """Return one row per order with joined dimensions and engineered features."""
    o = c["orders"].copy()
    o["OrderTS"] = pd.to_datetime(o["OrderDate"].dt.strftime("%Y-%m-%d") + " " + o["OrderTime"])
    o["hour"] = o["OrderTS"].dt.hour
    o["order_min"] = o["hour"] * 60 + o["OrderTS"].dt.minute
    o["dow"] = o["OrderDate"].dt.dayofweek
    o["month"] = o["OrderDate"].dt.to_period("M").astype(str)

    r = c["restaurants"].rename(columns={"City": "RCity", "Rating": "RestaurantRating", "Area": "RArea"})
    d = o.merge(r[["RestaurantID", "RestaurantName", "RCity", "RArea", "Cuisine", "RestaurantRating", "AverageCost",
                   "RestaurantType", "Latitude", "Longitude"]], on="RestaurantID", how="left")
    cu = c["customers"].rename(columns={"City": "CCity", "Age": "CustomerAge"})
    d = d.merge(cu[["CustomerID", "CCity", "CustomerAge", "Membership", "RegistrationDate", "Gender"]],
                on="CustomerID", how="left")
    dp = c["delivery_partners"].rename(columns={"Rating": "PartnerRating", "City": "PCity"})
    d = d.merge(dp[["DeliveryPartnerID", "VehicleType", "PartnerRating", "AverageDeliveryTime",
                    "CompletedDeliveries", "PCity"]], on="DeliveryPartnerID", how="left")

    # --- weather (mean of the readings for that city+date; many dates are repeated) ---
    w = c["weather"].groupby(["City", "Date"], as_index=False).agg(
        Rainfall=("Rainfall", "mean"), Temperature=("Temperature", "mean"), Humidity=("Humidity", "mean"),
        WeatherCondition=("WeatherCondition", lambda s: s.mode().iat[0]))
    d = d.merge(w, left_on=["RCity", "OrderDate"], right_on=["City", "Date"], how="left").drop(columns=["City", "Date"])
    d["HasWeather"] = d["Rainfall"].notna().astype(int)
    d = _attach_traffic(d, c["traffic"])
    d["HasTraffic"] = d["TrafficScore"].notna().astype(int)
    # unmatched city/date -> city median (flag columns above keep this transparent)
    for col in ["Rainfall", "Temperature", "Humidity", "TrafficScore", "AverageSpeed"]:
        d[col] = d[col].fillna(d.groupby("RCity")[col].transform("median")).fillna(d[col].median())
    d["WeatherCondition"] = d["WeatherCondition"].fillna("Unknown")

    # --- basket size ---
    oi = c["order_items"].groupby("OrderID").agg(BasketSize=("Quantity", "sum"), ItemLines=("OrderItemID", "count"))
    d = d.merge(oi, on="OrderID", how="left")
    d[["BasketSize", "ItemLines"]] = d[["BasketSize", "ItemLines"]].fillna(0)

    # --- feedback (post-delivery -> analysis only, never a delivery-time predictor) ---
    fb = c["customer_feedback"].assign(AvgRating=lambda x: x[["CustomerRating", "DeliveryRating", "FoodRating"]].mean(axis=1))
    d = d.merge(fb[["OrderID", "CustomerRating", "DeliveryRating", "FoodRating", "AvgRating", "Sentiment"]],
                on="OrderID", how="left")
    pay = c["payments"][["OrderID", "PaymentStatus"]]
    d = d.merge(pay, on="OrderID", how="left")

    # ------------------------------ engineered features ------------------------------
    d["PeakHour"] = (d["hour"].between(12, 13) | d["hour"].between(19, 21)).astype(int)
    d["WeekendOrder"] = (d["dow"] >= 5).astype(int)
    d["RainImpact"] = pd.cut(d["Rainfall"], [-1, 0.5, 10, 50, np.inf], labels=[0, 1, 2, 3]).astype(int)
    d["CustomerTenure"] = (d["OrderDate"] - d["RegistrationDate"]).dt.days
    d["IsDelivered"] = d["OrderStatus"].isin(COMPLETED)
    d["IsCancelled"] = (d["OrderStatus"] == "Cancelled").astype(int)
    d["IsLate"] = (d["OrderStatus"] == "Delivered Late").astype(int)
    d["HasCoupon"] = d["CouponCode"].notna().astype(int)
    d["DiscountPct"] = np.where(d["FoodCost"] > 0, d["Discount"] / d["FoodCost"], 0)

    # customer-level aggregates (use all orders; for ANALYSIS only - churn table uses a cutoff)
    d = d.sort_values(["CustomerID", "OrderTS"])
    g = d.groupby("CustomerID")
    d["AverageBasketValue"] = g["FinalAmount"].transform("mean")
    d["CustomerLifetimeValue"] = g["FinalAmount"].transform("sum")
    span = (g["OrderTS"].transform("max") - g["OrderTS"].transform("min")).dt.days.clip(lower=30)
    d["OrderFrequency"] = g["OrderID"].transform("count") / (span / 30.0)      # orders per 30 days

    # restaurant popularity within city
    rp = d.groupby(["RCity", "RestaurantID"]).agg(RestOrders=("OrderID", "count"), RestRevenue=("FinalAmount", "sum")).reset_index()
    rp["RestaurantPopularityRank"] = rp.groupby("RCity")["RestOrders"].rank(ascending=False, method="min")
    rp["RestaurantRevenueRank"] = rp.groupby("RCity")["RestRevenue"].rank(ascending=False, method="min")
    d = d.merge(rp, on=["RCity", "RestaurantID"], how="left")

    # delivery partner efficiency = partner's avg delivery time / city-wide average (<1 is faster)
    city_avg = c["delivery_partners"].groupby("City")["AverageDeliveryTime"].mean()
    d["DeliveryEfficiency"] = d["AverageDeliveryTime"] / d["PCity"].map(city_avg)

    # restaurant mean rating from feedback (restaurant-level AverageRating)
    d["RestaurantAvgFeedback"] = d.groupby("RestaurantID")["AvgRating"].transform("mean")

    # distance PROXY: restaurant to centroid of restaurants in its city.
    # (customers carry no coordinates, so true restaurant->customer distance is NOT derivable)
    cen = c["restaurants"].groupby("City")[["Latitude", "Longitude"]].mean()
    d["DistanceProxyKm"] = haversine_km(d["Latitude"], d["Longitude"],
                                        d["RCity"].map(cen["Latitude"]), d["RCity"].map(cen["Longitude"]))
    return d.sort_values("OrderID").reset_index(drop=True)


def build_churn_table(abt: pd.DataFrame, c: dict, horizon_days: int = 60, cutoff: str | None = None):
    """Customer-level churn table with a strict temporal split.

    cutoff = last order date - horizon (default). Features use orders <= cutoff.
    Label churn=1 if the customer places NO order in (cutoff, cutoff+horizon].
    Only customers with >=1 order on/before cutoff are included (otherwise no history).
    """
    end = abt["OrderDate"].max()
    cutoff = pd.Timestamp(cutoff) if cutoff else end - pd.Timedelta(days=horizon_days)
    hist = abt[abt["OrderDate"] <= cutoff]
    fut = abt[(abt["OrderDate"] > cutoff) & (abt["OrderDate"] <= cutoff + pd.Timedelta(days=horizon_days))]
    g = hist.groupby("CustomerID")
    f = pd.DataFrame({
        "Recency": (cutoff - g["OrderDate"].max()).dt.days,
        "Frequency": g["OrderID"].count(),
        "Monetary": g["FinalAmount"].sum(),
        "AvgOrderValue": g["FinalAmount"].mean(),
        "CancelRate": g["IsCancelled"].mean(),
        "LateRate": g["IsLate"].mean(),
        "CouponRate": g["HasCoupon"].mean(),
        "AvgDeliveryTime": g["DeliveryTimeMinutes"].mean(),
        "AvgRating": g["AvgRating"].mean(),
        "DistinctRestaurants": g["RestaurantID"].nunique(),
        "WeekendShare": g["WeekendOrder"].mean(),
        "FirstOrderAgeDays": (cutoff - g["OrderDate"].min()).dt.days,
    })
    f["AvgGapDays"] = (f["FirstOrderAgeDays"] - f["Recency"]) / (f["Frequency"] - 1).replace(0, np.nan)
    cu = c["customers"].set_index("CustomerID")
    f["Tenure"] = (cutoff - cu["RegistrationDate"].reindex(f.index)).dt.days
    f["Age"] = cu["Age"].reindex(f.index)
    f["Membership"] = cu["Membership"].reindex(f.index)
    f["PreferredCuisine"] = cu["PreferredCuisine"].reindex(f.index)
    f["AvgRating"] = f["AvgRating"].fillna(f["AvgRating"].median())
    f["AvgDeliveryTime"] = f["AvgDeliveryTime"].fillna(f["AvgDeliveryTime"].median())
    f["AvgGapDays"] = f["AvgGapDays"].fillna(-1)   # -1 = single-order customer (no gap observable)
    f["Churn"] = (~f.index.isin(fut["CustomerID"].unique())).astype(int)
    return f.reset_index(), cutoff


def build_all(c: dict):
    """Convenience: returns (abt, churn_table, cutoff)."""
    abt = build_order_base(c)
    churn, cutoff = build_churn_table(abt, c)
    return abt, churn, cutoff
