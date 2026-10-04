"""eda.py - Exploratory analysis: 20 Matplotlib + 18 Seaborn charts, KPI tables.

Revenue definition used everywhere: sum of FinalAmount over COMPLETED orders
(Delivered + Delivered Late). Cancelled / 'Food Not Delivered' orders earn nothing.
Late definition: OrderStatus == 'Delivered Late' (share of completed deliveries).
Run: python -m src.eda
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from .ingest import load_cleaned, ROOT
from .features import build_order_base

RED, DARK, GREY = "#E23744", "#2D2D2D", "#9C9C9C"
IMG = ROOT / "images"; OUT = ROOT / "data" / "cleaned"
sns.set_theme(style="whitegrid", rc={"axes.titleweight": "bold"})
COUNT = {"mpl": 0, "sns": 0}


def save(fig, name, kind):
    """Save a chart as images/eda_<kind>_<name>.png and count it."""
    fig.tight_layout(); fig.savefig(IMG / f"eda_{kind}_{name}.png", dpi=120); plt.close(fig); COUNT[kind] += 1


def fig_ax(w=7, h=4.2):
    return plt.subplots(figsize=(w, h))


def restaurant_scorecard(abt: pd.DataFrame) -> pd.DataFrame:
    """Composite restaurant score = 0.4*z(rating) - 0.3*z(cancel rate) - 0.3*z(late rate); min 5 orders."""
    g = abt.groupby(["RestaurantID", "RestaurantName", "RCity", "Cuisine"]).agg(
        Orders=("OrderID", "count"), Revenue=("FinalAmount", lambda s: s[abt.loc[s.index, "IsDelivered"]].sum()),
        Rating=("RestaurantRating", "first"), CancelRate=("IsCancelled", "mean"),
        LateRate=("IsLate", lambda s: s.sum() / max(abt.loc[s.index, "IsDelivered"].sum(), 1)),
        AvgDeliveryTime=("DeliveryTimeMinutes", "mean")).reset_index()
    g = g[g.Orders >= 5].copy(); z = lambda s: (s - s.mean()) / s.std()
    g["CompositeScore"] = 0.4 * z(g.Rating) - 0.3 * z(g.CancelRate) - 0.3 * z(g.LateRate)
    g["CityRank"] = g.groupby("RCity")["CompositeScore"].rank(ascending=False, method="first")
    g["CityRankFromBottom"] = g.groupby("RCity")["CompositeScore"].rank(ascending=True, method="first")
    return g.sort_values("CompositeScore", ascending=False)


def main():
    IMG.mkdir(exist_ok=True)
    c = load_cleaned(); abt = build_order_base(c)
    dl = abt[abt.IsDelivered].copy()
    dl["month"] = dl.OrderDate.dt.to_period("M").dt.to_timestamp()
    rng = np.random.RandomState(42)

    # ======================================================== MATPLOTLIB (20)
    s = dl.groupby("RCity").FinalAmount.sum().sort_values(ascending=False).head(12)
    f, a = fig_ax(8, 4.5); a.bar(s.index, s.values / 1e5, color=RED); a.set(title="Revenue by city (top 12)", xlabel="City", ylabel="Revenue (INR lakh)"); a.tick_params(axis="x", rotation=45); save(f, "01_revenue_by_city", "mpl")
    s = dl.groupby("Cuisine").OrderID.count().sort_values(ascending=False).head(12)
    f, a = fig_ax(8, 4.5); a.bar(s.index, s.values, color=DARK); a.set(title="Cuisine popularity (completed orders)", xlabel="Cuisine", ylabel="Orders"); a.tick_params(axis="x", rotation=45); save(f, "02_cuisine_popularity", "mpl")
    m = dl.groupby("month").FinalAmount.sum()
    f, a = fig_ax(8, 4); a.plot(m.index, m.values / 1e5, marker="o", color=RED, label="Monthly revenue"); a.set(title="Monthly revenue trend", xlabel="Month", ylabel="Revenue (INR lakh)"); a.legend(); save(f, "03_monthly_revenue", "mpl")
    m2 = dl.groupby("month").DeliveryTimeMinutes.mean()
    f, a = fig_ax(8, 4); a.plot(m2.index, m2.values, marker="s", color=DARK, label="Avg delivery time"); a.set(title="Average delivery time by month", xlabel="Month", ylabel="Minutes", ylim=(0, m2.max() * 1.3)); a.legend(); save(f, "04_delivery_time_trend", "mpl")
    smp = dl.dropna(subset=["DeliveryTimeMinutes"]).sample(2500, random_state=1)
    f, a = fig_ax(); a.scatter(smp.DistanceProxyKm, smp.DeliveryTimeMinutes, s=6, alpha=.4, color=RED, label="orders (sample 2,500)"); a.set(title="Delivery time vs distance proxy", xlabel="Restaurant distance from city centroid (km, proxy)", ylabel="Delivery time (min)"); a.legend(); save(f, "05_scatter_time_vs_distance", "mpl")
    f, a = fig_ax(); a.hist(dl.FinalAmount, bins=40, color=RED, label="orders"); a.set(title="Order value distribution", xlabel="FinalAmount (INR)", ylabel="Orders"); a.legend(); save(f, "06_hist_order_value", "mpl")
    f, a = fig_ax(); a.hist(dl.DeliveryTimeMinutes.dropna(), bins=40, color=DARK, label="completed orders"); a.axvline(dl.DeliveryTimeMinutes.median(), color=RED, ls="--", label="median"); a.set(title="Delivery time distribution", xlabel="Minutes", ylabel="Orders"); a.legend(); save(f, "07_hist_delivery_time", "mpl")
    pm = abt.PaymentMethod.value_counts(); f, a = fig_ax(5.5, 5.5); a.pie(pm, labels=pm.index, autopct="%1.1f%%", colors=sns.color_palette("Reds_r", len(pm))); a.set_title("Payment method share"); save(f, "08_pie_payment_method", "mpl")
    st = abt.OrderStatus.value_counts(); f, a = fig_ax(5.5, 5.5); a.pie(st, labels=st.index, autopct="%1.1f%%", colors=[RED, DARK, GREY, "#F4A6AD"]); a.set_title("Order status share"); save(f, "09_pie_order_status", "mpl")
    top_c = dl.groupby("RCity").FinalAmount.sum().nlargest(8).index; top_q = dl.Cuisine.value_counts().head(6).index
    pv = dl[dl.RCity.isin(top_c) & dl.Cuisine.isin(top_q)].pivot_table(index="RCity", columns="Cuisine", values="FinalAmount", aggfunc="sum", fill_value=0) / 1e3
    f, a = fig_ax(9, 5); pv.plot(kind="stacked" if False else "bar", stacked=True, ax=a, colormap="Reds"); a.set(title="Revenue by cuisine per city (top 8 cities x top 6 cuisines)", xlabel="City", ylabel="Revenue (INR '000)"); a.legend(title="Cuisine", fontsize=7); a.tick_params(axis="x", rotation=45); save(f, "10_stacked_revenue_cuisine_city", "mpl")
    cum = dl.groupby("OrderDate").FinalAmount.sum().cumsum() / 1e5
    f, a = fig_ax(8, 4); a.fill_between(cum.index, cum.values, color=RED, alpha=.5, label="Cumulative revenue"); a.set(title="Cumulative revenue growth", xlabel="Date", ylabel="INR lakh"); a.legend(); save(f, "11_area_cumulative_revenue", "mpl")
    vt = [g.DeliveryTimeMinutes.dropna().values for _, g in dl.groupby("VehicleType")]
    f, a = fig_ax(); a.boxplot(vt, labels=sorted(dl.VehicleType.unique())); a.set(title="Delivery time by vehicle type", xlabel="Vehicle", ylabel="Minutes"); save(f, "12_box_time_by_vehicle", "mpl")
    hw = abt.assign(wk=np.where(abt.WeekendOrder == 1, "Weekend", "Weekday")).groupby(["hour", "wk"]).OrderID.count().unstack()
    hw = hw.div(abt.groupby(np.where(abt.WeekendOrder == 1, "Weekend", "Weekday")).OrderDate.nunique())   # per-day average
    f, a = fig_ax(9, 4); hw.plot(kind="bar", ax=a, color=[RED, DARK]); a.set(title="Orders per day by hour: weekday vs weekend", xlabel="Hour of day", ylabel="Avg orders per day"); a.legend(title="Day type"); save(f, "13_orders_by_hour", "mpl")
    cr = abt.groupby("Cuisine").IsCancelled.mean().sort_values() * 100
    f, a = fig_ax(7, 6); a.barh(cr.index, cr.values, color=RED); a.set(title="Cancellation rate by cuisine", xlabel="Cancelled orders (%)", ylabel="Cuisine"); save(f, "14_cancellation_by_cuisine", "mpl")
    mb = dl.groupby("Membership").FinalAmount.mean()
    f, a = fig_ax(6, 4); a.bar(mb.index, mb.values, color=DARK); a.set(title="Average basket value by membership", xlabel="Membership", ylabel="Avg FinalAmount (INR)"); save(f, "15_basket_by_membership", "mpl")
    oc = abt.groupby("CustomerID").size(); rep = pd.Series({"One-time": (oc == 1).sum(), "Repeat": (oc > 1).sum()})
    f, a = fig_ax(5, 5); a.pie(rep, labels=rep.index, autopct="%1.1f%%", colors=[GREY, RED]); a.set_title("Repeat vs one-time customers"); save(f, "16_pie_repeat_customers", "mpl")
    tr = dl.groupby("RestaurantName").FinalAmount.sum().nlargest(10).iloc[::-1]
    f, a = fig_ax(8, 4.5); a.barh(tr.index, tr.values / 1e3, color=RED); a.set(title="Top 10 restaurants by revenue", xlabel="Revenue (INR '000)", ylabel="Restaurant"); save(f, "17_top_restaurants", "mpl")
    tl = dl.groupby("TrafficScore").DeliveryTimeMinutes.agg(["mean", "sem"])
    f, a = fig_ax(6, 4); a.bar(["Low", "Moderate", "High", "Severe"][:len(tl)], tl["mean"], yerr=1.96 * tl["sem"], color=DARK, capsize=4, label="mean +/- 95% CI"); a.set(title="Delivery time by traffic level", xlabel="Traffic level", ylabel="Minutes", ylim=(0, 50)); a.legend(); save(f, "18_time_by_traffic", "mpl")
    ps = abt.groupby(["PaymentMethod", "PaymentStatus"]).size().unstack(fill_value=0)
    f, a = fig_ax(8, 4.5); ps.plot(kind="bar", stacked=True, ax=a, colormap="Reds"); a.set(title="Payment status by method", xlabel="Payment method", ylabel="Orders"); a.legend(title="Status", fontsize=7); a.tick_params(axis="x", rotation=30); save(f, "19_payment_status", "mpl")
    cp = abt[abt.HasCoupon == 1].merge(c["promotions"][["CouponCode", "CampaignName"]], on="CouponCode").groupby("CampaignName").Discount.sum().sort_values()
    f, a = fig_ax(8, 5); a.barh(cp.index, cp.values / 1e3, color=RED); a.set(title="Discount value by campaign", xlabel="Total discount (INR '000)", ylabel="Campaign"); save(f, "20_discount_by_campaign", "mpl")

    # ======================================================== SEABORN (18)
    num = ["FinalAmount", "FoodCost", "DeliveryFee", "Discount", "DeliveryTimeMinutes", "TrafficScore", "RainImpact", "Rainfall",
           "Temperature", "BasketSize", "DistanceProxyKm", "DeliveryEfficiency", "PartnerRating", "RestaurantRating", "CustomerTenure", "PeakHour", "WeekendOrder"]
    f, a = fig_ax(11, 9); sns.heatmap(abt[num].corr(), cmap="RdBu_r", center=0, annot=True, fmt=".2f", annot_kws={"size": 6}, ax=a); a.set(title="Correlation heatmap of numeric features"); save(f, "01_corr_heatmap", "sns")
    pp = sns.pairplot(dl[["DeliveryTimeMinutes", "FinalAmount", "TrafficScore", "DistanceProxyKm", "BasketSize"]].dropna().sample(800, random_state=1), corner=True, plot_kws=dict(s=8, alpha=.4, color=RED), diag_kws=dict(color=RED))
    pp.figure.suptitle("Pairplot of key delivery/order features (sample 800)", y=1.01); pp.figure.savefig(IMG / "eda_sns_02_pairplot.png", dpi=110); plt.close(pp.figure); COUNT["sns"] += 1
    f, a = fig_ax(); sns.countplot(data=abt, x="OrderStatus", color=RED, ax=a); a.set(title="Order status counts", xlabel="Order status", ylabel="Orders"); a.tick_params(axis="x", rotation=15); save(f, "03_count_order_status", "sns")
    f, a = fig_ax(); sns.countplot(data=abt, y="PaymentMethod", color=DARK, ax=a); a.set(title="Payment method counts", xlabel="Orders", ylabel="Payment method"); save(f, "04_count_payment_method", "sns")
    t8 = dl.RCity.value_counts().head(8).index
    f, a = fig_ax(10, 4.5); sns.violinplot(data=dl[dl.RCity.isin(t8)], x="RCity", y="DeliveryTimeMinutes", color="#F4A6AD", ax=a); a.set(title="Delivery time by city (top 8 by orders)", xlabel="City", ylabel="Minutes"); a.tick_params(axis="x", rotation=30); save(f, "05_violin_time_by_city", "sns")
    rs = c["restaurants"].sample(600, random_state=1)
    f, a = fig_ax(10, 5); sns.stripplot(data=rs, x="Cuisine", y="Rating", size=3, color=RED, alpha=.6, ax=a); a.set(title="Restaurant rating by cuisine (sample 600)", xlabel="Cuisine", ylabel="Rating"); a.tick_params(axis="x", rotation=60); save(f, "06_strip_rating_by_cuisine", "sns")
    f, a = fig_ax(); sns.boxplot(data=dl, x="Membership", y="FinalAmount", color="#F4A6AD", ax=a); a.set(title="Basket value by membership tier", xlabel="Membership", ylabel="FinalAmount (INR)"); save(f, "07_box_basket_membership", "sns")
    sm = dl.dropna(subset=["DeliveryTimeMinutes"]).sample(3000, random_state=2)
    f, a = fig_ax(); sns.regplot(data=sm, x="DistanceProxyKm", y="DeliveryTimeMinutes", scatter_kws=dict(s=5, alpha=.3, color=RED), line_kws=dict(color=DARK), ax=a); a.set(title="Delivery time vs distance proxy (fit)", xlabel="Distance proxy (km)", ylabel="Minutes"); save(f, "08_reg_time_vs_distance", "sns")
    f, a = fig_ax(); sns.regplot(data=sm, x="TrafficScore", y="DeliveryTimeMinutes", x_jitter=.15, scatter_kws=dict(s=5, alpha=.3, color=RED), line_kws=dict(color=DARK), ax=a); a.set(title="Delivery time vs traffic score (fit)", xlabel="Traffic score (1=Low, 4=Severe)", ylabel="Minutes"); save(f, "09_reg_time_vs_traffic", "sns")
    f, a = fig_ax(); sns.kdeplot(data=dl.dropna(subset=["DeliveryTimeMinutes"]), x="DeliveryTimeMinutes", hue="OrderStatus", fill=True, common_norm=False, ax=a); a.set(title="Delivery time density: on-time vs late", xlabel="Minutes", ylabel="Density"); save(f, "10_kde_delivery_time", "sns")
    f, a = fig_ax(); sns.kdeplot(data=abt, x="FoodCost", fill=True, color=RED, label="FoodCost", ax=a); a.set(title="Food cost density", xlabel="Food cost (INR)", ylabel="Density"); a.legend(); save(f, "11_kde_food_cost", "sns")
    hm = abt.groupby(["dow", "hour"]).size().unstack(fill_value=0); hm.index = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    f, a = fig_ax(11, 4); sns.heatmap(hm, cmap="Reds", ax=a, cbar_kws={"label": "Orders"}); a.set(title="Order volume by weekday and hour", xlabel="Hour", ylabel="Weekday"); save(f, "12_heatmap_hour_dow", "sns")
    f, a = fig_ax(); sns.barplot(data=abt, x="WeatherCondition", y="IsCancelled", color=RED, ax=a, errorbar=("ci", 95)); a.set(title="Cancellation rate by weather condition", xlabel="Weather", ylabel="Cancellation rate"); a.tick_params(axis="x", rotation=30); save(f, "13_bar_cancel_by_weather", "sns")
    f, a = fig_ax(); sns.boxplot(data=dl, x="RainImpact", y="DeliveryTimeMinutes", color="#F4A6AD", ax=a); a.set(title="Delivery time by rain impact (0=none..3=heavy)", xlabel="RainImpact", ylabel="Minutes"); save(f, "14_box_time_by_rain", "sns")
    clv = abt.groupby(["CustomerID"]).agg(CLV=("CustomerLifetimeValue", "first"), City=("CCity", "first")); top_cc = clv.City.value_counts().head(8).index
    f, a = fig_ax(10, 4.5); sns.boxenplot(data=clv[clv.City.isin(top_cc)], x="City", y="CLV", color="#F4A6AD", ax=a); a.set(title="Customer lifetime value by customer city (top 8)", xlabel="City", ylabel="CLV (INR)"); a.tick_params(axis="x", rotation=30); save(f, "15_boxen_clv_by_city", "sns")
    fb = abt.dropna(subset=["Sentiment"])
    f, a = fig_ax(); sns.countplot(data=fb, x="OrderStatus", hue="Sentiment", palette=["#2D2D2D", "#F4A6AD", RED], ax=a); a.set(title="Feedback sentiment by order status", xlabel="Order status", ylabel="Reviews"); a.tick_params(axis="x", rotation=15); save(f, "16_count_sentiment_by_status", "sns")
    hc = dl.pivot_table(index="RCity", columns="Cuisine", values="FinalAmount", aggfunc="sum", fill_value=0).loc[lambda x: x.sum(1).nlargest(12).index, lambda x: x.sum().nlargest(10).index] / 1e3
    f, a = fig_ax(10, 5.5); sns.heatmap(hc, cmap="Reds", annot=True, fmt=".0f", annot_kws={"size": 6}, ax=a, cbar_kws={"label": "INR '000"}); a.set(title="Revenue: top cities x top cuisines", xlabel="Cuisine", ylabel="City"); save(f, "17_heatmap_city_cuisine", "sns")
    ms = abt.assign(month=abt.OrderDate.dt.to_period("M").dt.to_timestamp()).groupby(["month", "OrderStatus"]).size().reset_index(name="orders")
    f, a = fig_ax(9, 4.5); sns.lineplot(data=ms, x="month", y="orders", hue="OrderStatus", marker="o", ax=a); a.set(title="Monthly orders by status", xlabel="Month", ylabel="Orders"); save(f, "18_line_orders_by_status", "sns")

    # ======================================================== KPI tables
    sc = restaurant_scorecard(abt); sc.to_csv(OUT / "restaurant_scorecard.csv", index=False)
    top10 = sc[sc.CityRank <= 10]; bot10 = sc[sc.CityRankFromBottom <= 10]
    top10.to_csv(OUT / "restaurants_top10_per_city.csv", index=False); bot10.to_csv(OUT / "restaurants_bottom10_per_city.csv", index=False)
    comp = abt[abt.IsDelivered]
    cust_orders = abt.groupby("CustomerID").OrderID.count()
    kpi = dict(
        orders_total=int(len(abt)), customers_with_orders=int(abt.CustomerID.nunique()), restaurants_with_orders=int(abt.RestaurantID.nunique()),
        revenue_inr=float(comp.FinalAmount.sum()), avg_order_value=float(comp.FinalAmount.mean()),
        completed_orders=int(len(comp)), cancelled_pct=float(abt.IsCancelled.mean() * 100),
        food_not_delivered_pct=float((abt.OrderStatus == "Food Not Delivered").mean() * 100),
        late_pct_of_completed=float(comp.IsLate.mean() * 100),
        avg_delivery_min=float(comp.DeliveryTimeMinutes.mean()), median_delivery_min=float(comp.DeliveryTimeMinutes.median()),
        p90_delivery_min=float(np.percentile(comp.DeliveryTimeMinutes.dropna(), 90)), p95_delivery_min=float(np.percentile(comp.DeliveryTimeMinutes.dropna(), 95)),
        avg_feedback_rating=float(abt.AvgRating.mean()), one_time_customers_pct=float((cust_orders == 1).mean() * 100),
        top_city=str(comp.groupby("RCity").FinalAmount.sum().idxmax()), top_cuisine=str(comp.groupby("Cuisine").FinalAmount.sum().idxmax()),
        peak_hour=int(abt.groupby("hour").size().idxmax()), weekend_orders_pct=float(abt.WeekendOrder.mean() * 100),
        avg_basket_by_membership=comp.groupby("Membership").FinalAmount.mean().round(1).to_dict(),
        delivery_time_by_traffic=comp.groupby("TrafficScore").DeliveryTimeMinutes.mean().round(2).to_dict(),
        delivery_time_by_rain=comp.groupby("RainImpact").DeliveryTimeMinutes.mean().round(2).to_dict(),
        late_pct_by_traffic=(comp.groupby("TrafficScore").IsLate.mean() * 100).round(2).to_dict(),
        cancel_pct_by_weather=(abt.groupby("WeatherCondition").IsCancelled.mean() * 100).round(2).to_dict(),
        cancel_pct_by_payment=(abt.groupby("PaymentMethod").IsCancelled.mean() * 100).round(2).to_dict(),
        cancel_pct_by_cuisine=(abt.groupby("Cuisine").IsCancelled.mean() * 100).round(2).sort_values().to_dict(),
        weather_join_rate=float(abt.HasWeather.mean()), traffic_join_rate=float(abt.HasTraffic.mean()),
        revenue_by_city=comp.groupby("RCity").FinalAmount.sum().round(0).sort_values(ascending=False).head(5).to_dict(),
        revenue_by_cuisine=comp.groupby("Cuisine").FinalAmount.sum().round(0).sort_values(ascending=False).head(5).to_dict(),
        coupon_usage_pct=float(abt.HasCoupon.mean() * 100), avg_discount_pct_when_coupon=float(abt.loc[abt.HasCoupon == 1, "DiscountPct"].mean() * 100),
        payment_failed_pct=float((abt.PaymentStatus == "Failed").mean() * 100),
        cancel_rate_late_vs_traffic_corr=float(abt[["IsCancelled", "TrafficScore"]].corr().iloc[0, 1]),
        mpl_charts=COUNT["mpl"], sns_charts=COUNT["sns"])
    # statistical significance checks used in the insights report (no scipy: use normal approx / ANOVA via numpy)
    a, b = comp[comp.TrafficScore >= 3].DeliveryTimeMinutes.dropna(), comp[comp.TrafficScore <= 2].DeliveryTimeMinutes.dropna()
    se = np.sqrt(a.var() / len(a) + b.var() / len(b)); kpi["traffic_high_vs_low_diff_min"] = float(a.mean() - b.mean()); kpi["traffic_high_vs_low_z"] = float((a.mean() - b.mean()) / se)
    a, b = comp[comp.RainImpact >= 2].DeliveryTimeMinutes.dropna(), comp[comp.RainImpact <= 1].DeliveryTimeMinutes.dropna()
    se = np.sqrt(a.var() / len(a) + b.var() / len(b)); kpi["rain_heavy_vs_light_diff_min"] = float(a.mean() - b.mean()); kpi["rain_heavy_vs_light_z"] = float((a.mean() - b.mean()) / se)
    with open(ROOT / "reports" / "kpis.json", "w") as fh:
        json.dump(kpi, fh, indent=2, default=str)
    print(f"charts: matplotlib={COUNT['mpl']} seaborn={COUNT['sns']}")
    print(json.dumps({k: v for k, v in kpi.items() if not isinstance(v, dict)}, indent=1, default=str))


if __name__ == "__main__":
    main()
