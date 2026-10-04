"""dashboard_preview.py - Python-rendered PREVIEW of the six Power BI pages (images/powerbi_preview_*.png).

These are NOT the .pbix; they show what each page should contain, computed from the same
CSVs in powerbi/data/. Build the real dashboard in Power BI Desktop using powerbi/README_BUILD.md.
Run: python -m src.dashboard_preview
"""
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .ingest import ROOT

D = ROOT / "powerbi" / "data"; IMG = ROOT / "images"
RED, INK, MID = "#E23744", "#2D2D2D", "#F4A6AD"


def rd(n, **kw): return pd.read_csv(D / f"{n}.csv", **kw)


def canvas(title, nrows=2, ncols=3):
    fig = plt.figure(figsize=(14, 7.8)); fig.patch.set_facecolor("white")
    fig.suptitle(f"{title}   [Python preview of Power BI page]", x=0.02, ha="left", fontsize=15, fontweight="bold", color=INK)
    return fig


def cards(fig, items, y=0.84):
    n = len(items); w = 0.94 / n
    for i, (lab, val) in enumerate(items):
        ax = fig.add_axes([0.03 + i * w, y, w - 0.012, 0.09]); ax.set_facecolor("#FDECEE"); ax.set_xticks([]); ax.set_yticks([])
        for sp in ax.spines.values(): sp.set_visible(False)
        ax.text(0.05, 0.58, val, fontsize=18, fontweight="bold", color=RED, transform=ax.transAxes); ax.text(0.05, 0.12, lab, fontsize=9, color=INK, transform=ax.transAxes)


def grid(fig, n, **_):
    """3 charts -> one tall row; 4 charts -> 2x2. Rounded margins leave room for labels."""
    if n == 3:
        return [fig.add_axes([0.05 + i * 0.32, 0.17, 0.26, 0.52]) for i in range(3)]
    return [fig.add_axes([0.06 + (i % 2) * 0.48, 0.47 - (i // 2) * 0.38, 0.41, 0.27]) for i in range(4)]


def main():
    f = rd("fact_orders", parse_dates=["OrderDate"]); cu = rd("dim_customer", parse_dates=["RegistrationDate", "FirstOrderDate", "LastOrderDate"])
    dl = f[f.IsDelivered == 1].copy(); dl["m"] = dl.OrderDate.dt.to_period("M").dt.to_timestamp()

    # 1 Executive
    fig = canvas("1. Executive Dashboard"); active = int(((cu.DaysSinceLastOrder <= 60)).sum())
    cards(fig, [("Total revenue", f"INR {dl.Revenue.sum()/1e5:.1f}L"), ("Total orders", f"{len(f):,}"), ("Avg delivery (min)", f"{dl.DeliveryTimeMinutes.mean():.1f}"), ("Avg rating", f"{f.AvgRating.mean():.2f}"), ("Active customers (60d)", f"{active:,}"), ("% late deliveries", f"{dl.IsLate.mean()*100:.1f}%")])
    a = grid(fig, 3); m = dl.groupby("m").Revenue.sum() / 1e5
    a[0].plot(m.index, m.values, color=RED, marker="o"); a[0].set_title("Monthly revenue (INR lakh)", fontsize=10)
    c = dl.groupby("City").Revenue.sum().nlargest(8).iloc[::-1] / 1e5; a[1].barh(c.index, c.values, color=RED); a[1].set_title("Revenue by city (top 8)", fontsize=10)
    s = f.OrderStatus.value_counts(); a[2].pie(s, labels=s.index, autopct="%1.0f%%", colors=[INK, MID, RED, "#9C9C9C"], textprops={"fontsize": 8}); a[2].set_title("Order status", fontsize=10)
    fig.text(0.03, 0.02, "Slicers: City, Date range", fontsize=9, color="#777"); [x.tick_params(axis='x', labelrotation=30, labelsize=8) for x in fig.axes if x.get_xlabel() == '' and x.get_title()]; fig.savefig(IMG / "powerbi_preview_1_executive.png", dpi=90); plt.close(fig)

    # 2 Customer
    fig = canvas("2. Customer Analytics"); ordering = cu[cu.Orders > 0]
    cards(fig, [("Retention (repeat share)", f"{(ordering.Orders >= 2).mean()*100:.1f}%"), ("One-time customers", f"{(ordering.Orders == 1).mean()*100:.1f}%"), ("Never ordered", f"{(cu.Orders == 0).mean()*100:.1f}%"), ("Median CLV", f"INR {ordering.CLV.median():.0f}"), ("Avg orders / customer", f"{ordering.Orders.mean():.2f}")])
    a = grid(fig, 3); sg = cu.Segment.value_counts().reindex(["New", "Active", "At-Risk", "Lapsed", "No orders"]); a[0].bar(sg.index, sg.values, color=RED); a[0].set_title("Segments (as of 31-Dec-2024)", fontsize=10); a[0].tick_params(axis="x", rotation=30)
    g = cu.groupby(cu.RegistrationDate.dt.to_period("Q").dt.to_timestamp()).size().cumsum(); a[1].plot(g.index, g.values, color=INK); a[1].set_title("Customer growth (cumulative registrations)", fontsize=10)
    a[2].hist(ordering.CLV, bins=40, color=RED); a[2].set_title("CLV distribution (INR)", fontsize=10)
    [x.tick_params(axis='x', labelrotation=30, labelsize=8) for x in fig.axes if x.get_xlabel() == '' and x.get_title()]; fig.savefig(IMG / "powerbi_preview_2_customer.png", dpi=90); plt.close(fig)

    # 3 Restaurant
    fig = canvas("3. Restaurant Analytics"); sc = pd.read_csv(ROOT / "data" / "cleaned" / "restaurant_scorecard.csv"); rs = rd("dim_restaurant")
    cards(fig, [("Restaurants scored", f"{len(sc):,}"), ("Median orders / restaurant", f"{sc.Orders.median():.0f}"), ("Avg composite score", f"{sc.CompositeScore.mean():.2f}"), ("Avg rating", f"{rs.Rating.mean():.2f}")])
    a = grid(fig, 3); t = sc.nlargest(10, "CompositeScore").iloc[::-1]; a[0].barh(t.RestaurantName + " (" + t.RCity + ")", t.CompositeScore, color=RED); a[0].set_title("Top 10 by composite score", fontsize=10); a[0].tick_params(labelsize=7)
    b = sc.nsmallest(10, "CompositeScore").iloc[::-1]; a[1].barh(b.RestaurantName + " (" + b.RCity + ")", b.CompositeScore, color=INK); a[1].set_title("Bottom 10 (watch-list)", fontsize=10); a[1].tick_params(labelsize=7)
    cr = dl.merge(rs[["RestaurantID", "Cuisine"]], on="RestaurantID").groupby("Cuisine").Revenue.sum().nlargest(10).iloc[::-1] / 1e5; a[2].barh(cr.index, cr.values, color=RED); a[2].set_title("Cuisine revenue (INR lakh)", fontsize=10)
    [x.tick_params(axis='x', labelrotation=30, labelsize=8) for x in fig.axes if x.get_xlabel() == '' and x.get_title()]; fig.savefig(IMG / "powerbi_preview_3_restaurant.png", dpi=90); plt.close(fig)

    # 4 Delivery
    fig = canvas("4. Delivery Analytics"); cp = dl.groupby("DeliveryPartnerID").DeliveryTimeMinutes.agg(["mean", "count"]).query("count >= 10").nsmallest(10, "mean")
    cards(fig, [("Avg delivery (min)", f"{dl.DeliveryTimeMinutes.mean():.1f}"), ("P90 (min)", f"{np.percentile(dl.DeliveryTimeMinutes.dropna(), 90):.0f}"), ("% late", f"{dl.IsLate.mean()*100:.1f}%"), ("Partners with 10+ deliveries", f"{(dl.groupby('DeliveryPartnerID').size() >= 10).sum()}")])
    a = grid(fig, 4)
    a[0].barh(cp.index.astype(str).map(lambda x: f"Partner {x}"), cp["mean"], color=RED); a[0].invert_yaxis(); a[0].set_title("Fastest partners (min 10 deliveries)", fontsize=10)
    mt = dl.groupby("m").DeliveryTimeMinutes.mean(); a[1].plot(mt.index, mt.values, color=INK); a[1].set_ylim(0, 50); a[1].set_title("Delivery time trend (min)", fontsize=10)
    tt = dl.groupby("TrafficScore").DeliveryTimeMinutes.mean(); a[2].bar(["Low", "Moderate", "High", "Severe"][:len(tt)], tt.values, color=RED); a[2].set_ylim(0, 50); a[2].set_title("Traffic impact (mean min)", fontsize=10)
    rr = dl.groupby("RainImpact").DeliveryTimeMinutes.mean(); a[3].bar(["None", "Light", "Moderate", "Heavy"][:len(rr)], rr.values, color=INK); a[3].set_ylim(0, 50); a[3].set_title("Rain impact (mean min)", fontsize=10)
    [x.tick_params(axis='x', labelrotation=30, labelsize=8) for x in fig.axes if x.get_xlabel() == '' and x.get_title()]; fig.savefig(IMG / "powerbi_preview_4_delivery.png", dpi=90); plt.close(fig)

    # 5 Sales
    fig = canvas("5. Sales Dashboard"); pr = rd("dim_promotion"); cpn = dl[dl.HasCoupon == 1].merge(pr[["CouponCode", "CampaignName"]], on="CouponCode")
    cards(fig, [("Revenue", f"INR {dl.Revenue.sum()/1e5:.1f}L"), ("Discounts given", f"INR {dl.Discount.sum()/1e5:.1f}L"), ("Coupon usage", f"{f.HasCoupon.mean()*100:.0f}%"), ("Avg order value", f"INR {dl.FinalAmount.mean():.0f}")])
    a = grid(fig, 3); d = dl.groupby("OrderDate").Revenue.sum().rolling(30).mean() / 1e3; a[0].plot(d.index, d.values, color=RED); a[0].set_title("Daily revenue, 30-day average (INR '000)", fontsize=10)
    cc = cpn.groupby("CampaignName").Discount.sum().sort_values().tail(8) / 1e3; a[1].barh(cc.index, cc.values, color=RED); a[1].set_title("Discount by campaign (INR '000)", fontsize=10); a[1].tick_params(labelsize=7)
    pm = f.PaymentMethod.value_counts(); a[2].pie(pm, labels=pm.index, autopct="%1.0f%%", colors=plt.cm.Reds(np.linspace(.9, .3, len(pm))), textprops={"fontsize": 8}); a[2].set_title("Payment method", fontsize=10)
    [x.tick_params(axis='x', labelrotation=30, labelsize=8) for x in fig.axes if x.get_xlabel() == '' and x.get_title()]; fig.savefig(IMG / "powerbi_preview_5_sales.png", dpi=90); plt.close(fig)

    # 6 ML
    fig = canvas("6. ML Dashboard"); pa = rd("ml_delivery_predictions"); di = rd("ml_delivery_feature_importance"); cs = rd("ml_churn_scores"); ci = rd("ml_churn_feature_importance")
    dm = rd("ml_delivery_model_comparison"); cmod = rd("ml_churn_model_comparison")
    cards(fig, [("Delivery model test R²", f"{dm[dm.Stage.isin(['default', 'tuned'])].sort_values('CV_R2_mean').iloc[-1].R2:.3f}"), ("RMSE (min)", f"{dm[dm.Stage.isin(['default', 'tuned'])].sort_values('CV_R2_mean').iloc[-1].RMSE:.1f}"), ("Churn ROC-AUC", f"{cmod[cmod.Stage.isin(['default', 'tuned'])].sort_values('CV_AUC').iloc[-1].ROC_AUC:.2f}"), ("Churn base rate", f"{cs.Churn.mean()*100:.0f}%")])
    a = grid(fig, 4)
    a[0].scatter(pa.Predicted, pa.Actual, s=4, alpha=.3, color=RED); a[0].plot([0, 80], [0, 80], "k--"); a[0].set_title("Predicted vs actual (test)", fontsize=10); a[0].set_xlabel("Predicted"); a[0].set_ylabel("Actual")
    t = di.head(8).iloc[::-1]; a[1].barh(t.feature, t.permutation_mse_increase, color=INK); a[1].set_title("Feature importance (noise floor - see report)", fontsize=10); a[1].tick_params(labelsize=7)
    rk = cs[cs.Split == "test"].groupby("RiskSegment").Churn.agg(["count", "mean"]).reindex(["Low", "Medium", "High"]); a[2].bar(rk.index, rk["count"], color=RED); a[2].set_title("Test customers by predicted risk segment", fontsize=10)
    a[3].bar(rk.index, rk["mean"] * 100, color=INK); a[3].set_ylim(0, 100); a[3].set_title("Actual churn rate by segment, test set (%)", fontsize=10)
    [x.tick_params(axis='x', labelrotation=30, labelsize=8) for x in fig.axes if x.get_xlabel() == '' and x.get_title()]; fig.savefig(IMG / "powerbi_preview_6_ml.png", dpi=90); plt.close(fig)
    print("previews written")


if __name__ == "__main__":
    main()
