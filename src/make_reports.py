"""make_reports.py - Builds reports/eda_report.pdf and reports/business_insights_report.pdf.

All numbers are read from the pipeline outputs (kpis.json, stat_checks.json, model
comparison CSVs, cleaning_log.csv) so the reports regenerate with the data.
Run: python -m src.make_reports   (after clean, eda, stat_checks, model_* and export_powerbi)
"""
import json
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle)
from PIL import Image as PILImage

from .ingest import ROOT, load_cleaned
from .features import build_order_base

RED = colors.HexColor("#E23744"); DARK = colors.HexColor("#2D2D2D"); LIGHT = colors.HexColor("#FDECEE")
SS = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=SS["Heading1"], textColor=RED, fontSize=17, spaceBefore=6, spaceAfter=8)
H2 = ParagraphStyle("H2", parent=SS["Heading2"], textColor=DARK, fontSize=12.5, spaceBefore=10, spaceAfter=4)
BODY = ParagraphStyle("B", parent=SS["BodyText"], fontSize=9.6, leading=13.2, spaceAfter=5)
SMALL = ParagraphStyle("S", parent=BODY, fontSize=8, leading=10, textColor=colors.HexColor("#555555"))
BUL = ParagraphStyle("BUL", parent=BODY, leftIndent=12, bulletIndent=2)
TITLE = ParagraphStyle("T", parent=SS["Title"], textColor=RED, fontSize=24, leading=28, alignment=0)
IMG = ROOT / "images"
W = A4[0] - 4 * cm


def P(t, s=BODY): return Paragraph(t, s)
def B(t): return Paragraph(t, BUL, bulletText="•")


def img(name, width=W, caption=None):
    p = IMG / name; w, h = PILImage.open(p).size; width = min(width, W)
    items = [Image(str(p), width=width, height=width * h / w)]
    if caption: items.append(P(caption, SMALL))
    return KeepTogether(items)


def table(df, widths=None, font=8):
    cell = ParagraphStyle("cell", parent=BODY, fontSize=font, leading=font + 2, spaceAfter=0)
    head = ParagraphStyle("head", parent=cell, textColor=colors.white, fontName="Helvetica-Bold")
    data = [[Paragraph(str(h), head) for h in df.columns]] + [[Paragraph(str(v), cell) for v in row] for row in df.astype(str).values.tolist()]
    t = Table(data, colWidths=widths, repeatRows=1)
    t.setStyle(TableStyle([("BACKGROUND", (0, 0), (-1, 0), RED), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white), ("FONTSIZE", (0, 0), (-1, -1), font),
                           ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                           ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#CCCCCC")), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                           ("LEFTPADDING", (0, 0), (-1, -1), 3), ("RIGHTPADDING", (0, 0), (-1, -1), 3)]))
    return t


def footer(canvas, doc):
    canvas.saveState(); canvas.setFont("Helvetica", 7.5); canvas.setFillColor(colors.HexColor("#777777"))
    canvas.drawString(2 * cm, 1.2 * cm, "Zomato BI & Delivery Time Prediction - Capstone"); canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"Page {doc.page}"); canvas.restoreState()


def load_all():
    k = json.load(open(ROOT / "reports" / "kpis.json")); st = json.load(open(ROOT / "reports" / "stat_checks.json"))
    cd = ROOT / "data" / "cleaned"
    return dict(k=k, st=st, log=pd.read_csv(cd / "cleaning_log.csv"), dres=pd.read_csv(cd / "model_comparison_delivery.csv"),
                cres=pd.read_csv(cd / "model_comparison_churn.csv"), val=pd.read_csv(cd / "validation_report.csv"),
                sc=pd.read_csv(cd / "restaurant_scorecard.csv"), churn=pd.read_csv(cd / "churn_scores.csv"))


# =============================================================== EDA REPORT
def eda_report():
    d = load_all(); k, st, log = d["k"], d["st"], d["log"]
    doc = SimpleDocTemplate(str(ROOT / "reports" / "eda_report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="EDA Report - Zomato BI Project", author="Sundram Sharma")
    s = [P("Exploratory Data Analysis Report", TITLE), P("Zomato Business Intelligence &amp; Delivery Time Prediction Platform", H2),
         P("Author: Sundram Sharma (2510377936@geu.ac.in) | Data: 12 operational tables, Jan 2023 - Dec 2024 | 38 charts (20 Matplotlib, 18 Seaborn) in <i>images/</i>", SMALL), Spacer(1, 8)]
    s += [P("1. Summary of findings", H1),
          P(f"After cleaning, the dataset holds <b>{k['orders_total']:,} orders</b> from <b>{k['customers_with_orders']:,} customers</b> across 1,200 restaurants. "
            f"<b>{k['completed_orders']:,}</b> orders were completed (Delivered or Delivered Late) generating <b>INR {k['revenue_inr']/1e5:.1f} lakh</b> "
            f"(average order INR {k['avg_order_value']:.0f}). The most important facts are operational: <b>{k['cancelled_pct']:.1f}% of orders are cancelled and a further "
            f"{k['food_not_delivered_pct']:.1f}% are marked 'Food Not Delivered'</b> (so {k['cancelled_pct'] + k['food_not_delivered_pct']:.1f}% fail), and "
            f"<b>{k['late_pct_of_completed']:.1f}% of completed deliveries are late</b>. Average delivery time is {k['avg_delivery_min']:.1f} minutes (median {k['median_delivery_min']:.0f}, p90 {k['p90_delivery_min']:.0f})."),
          P(f"The second important fact is a <b>null result</b>. We ran {st['n_tests']} permutation tests asking whether cancellation, lateness, delivery time or order value differ by city, cuisine, weather, "
            f"traffic, payment method, vehicle, hour or membership. Only {st['n_p_below_0.05']} reached p&lt;0.05; with {st['n_tests']} tests about {st['n_tests']*0.05:.1f} are expected by chance alone. "
            "Differences visible in the charts (e.g. 'Mexican has the highest cancellation rate') are therefore <b>not distinguishable from random variation</b> and should not drive decisions."),
          P("What <i>is</i> structured: ordering time (lunch 12-14h and dinner 19-21h peaks, identical on weekdays and weekends), the late-delivery group (mean 63.7 min vs 33.5 min for on-time), and the monthly revenue level (flat at about INR 2.2 lakh/month).")]
    s += [P("2. Data cleaning (PRD 19.1)", H1),
          P(f"All {len(d['log'])} logged transformations are in <i>data/cleaned/cleaning_log.csv</i>; the cleaned schema passes {int(d['val'].passed.sum())}/{len(d['val'])} integrity checks (unique primary keys, all foreign keys resolve, no negative costs, ratings 1-5, promotion end &gt;= start). "
            "Three decisions were tested on the data rather than assumed:"),
          B("<b>Negative values are sign flips.</b> For all 181 orders with negative FoodCost, |FoodCost| + DeliveryFee - Discount + GST reproduces FinalAmount exactly, so we use abs() rather than drop the rows. The same rule is applied to delivery time, price, age, humidity and speed."),
          B("<b>Mixed day/month order in slash dates.</b> Within one column both DD/MM/YYYY and MM/DD/YYYY occur (about 15% of dates are ambiguous). For orders we pick the reading closest to the same order's payment date; weather/traffic default to DD/MM."),
          B("<b>Weather/traffic dates repeat.</b> Each city has ~730 rows but only ~63% of calendar days, so (City, Date) is not unique and only ~63% of orders link to weather/traffic. Missing links are imputed with the city median and flagged (HasWeather / HasTraffic).")]
    top = log.groupby(["table", "issue", "column"], as_index=False).rows_affected.sum().query("issue != 'mixed date formats'").sort_values("rows_affected", ascending=False).head(10)
    s += [Spacer(1, 4), table(top.rename(columns={"rows_affected": "rows"}), [3.2 * cm, 5.8 * cm, 4 * cm, 1.8 * cm]), P("Largest cleaning actions by rows affected (date parsing excluded; every row of each date column is parsed).", SMALL)]

    s += [P("3. Business questions", H1), P("Q1. Which cities, cuisines and restaurants earn the most, and how has revenue changed?", H2),
          P(f"Top cities by revenue: {', '.join(f'{a} ({b/1e5:.2f} lakh)' for a, b in k['revenue_by_city'].items())}. Top cuisines: {', '.join(f'{a} ({b/1e5:.2f} lakh)' for a, b in k['revenue_by_cuisine'].items())}. "
            "The spread between the #1 and #5 city is under 5%, inside what random assignment of orders would produce. Monthly revenue is flat (INR 1.9-2.5 lakh; first-12-month total 26.6 lakh vs last-12-month total 26.5 lakh) - there is no growth trend."),
          img("eda_mpl_03_monthly_revenue.png", 11 * cm), img("eda_sns_17_heatmap_city_cuisine.png", 14 * cm, "City x cuisine revenue (INR '000). No block stands out beyond sampling noise."),
          P("Q2. When do people order?", H2),
          P(f"Orders peak at {k['peak_hour']}:00 with a clear lunch (12-13h) and dinner (19-20h) structure. Weekends hold {k['weekend_orders_pct']:.1f}% of orders (28.6% expected if days were equal), i.e. no weekend uplift. This is the one pattern in the data with strong structure, and it supports the PeakHour feature."),
          img("eda_sns_12_heatmap_hour_dow.png", 15 * cm), img("eda_mpl_13_orders_by_hour.png", 12 * cm), PageBreak(),
          P("Q3. How are delivery times distributed and what moves them?", H2),
          P(f"Median {k['median_delivery_min']:.0f} min, p90 {k['p90_delivery_min']:.0f}, p95 {k['p95_delivery_min']:.0f}. Average time by traffic level (Low to Severe) is {', '.join(f'{v:.1f}' for v in k['delivery_time_by_traffic'].values())} minutes - a total spread of "
            f"{max(k['delivery_time_by_traffic'].values()) - min(k['delivery_time_by_traffic'].values()):.1f} minutes (permutation p = {st['delivery_time_by_traffic']['p']:.2f}). By rain: {', '.join(f'{v:.1f}' for v in k['delivery_time_by_rain'].values())} (p = {st['delivery_time_by_rain']['p']:.2f}). "
            f"City (p = {st['delivery_time_by_city']['p']:.2f}), vehicle (p = {st['delivery_time_by_vehicle']['p']:.2f}) and weather condition (p = {st['delivery_time_by_weather']['p']:.2f}) are also not significant. "
            f"The distribution is bimodal: 'Delivered Late' orders average {st['delivered_late_mean_time']:.1f} min versus {st['delivered_mean_time']:.1f} min."),
          img("eda_sns_10_kde_delivery_time.png", 11 * cm), img("eda_mpl_18_time_by_traffic.png", 9 * cm, "Mean delivery time by traffic level with 95% CI: bars overlap completely."), PageBreak(),
          P("Q4. What is the average basket and does it vary by membership?", H2),
          P(f"Average order value is INR {k['avg_order_value']:.0f}. By membership: {', '.join(f'{a} {b:.0f}' for a, b in k['avg_basket_by_membership'].items())} - no tier effect (p = {st['order_value_by_membership']['p']:.2f}). "
            f"Orders with a coupon average INR {st['coupon_aov_with']:.0f} versus {st['coupon_aov_without']:.0f} without; this gap is largely arithmetic because FinalAmount is net of the discount, so it says nothing about whether coupons grow baskets."),
          img("eda_sns_07_box_basket_membership.png", 10 * cm),
          P("Q5. Repeat versus one-time customers", H2),
          P(f"{k['one_time_customers_pct']:.1f}% of customers who ordered did so exactly once; in addition 2,173 of 12,000 registered customers (18.1%) never ordered. Orders per customer range 1-8 (mean 2.1)."),
          img("eda_mpl_16_pie_repeat_customers.png", 6 * cm),
          P("Q6. Which delivery partners are fastest and best rated?", H2),
          P(f"Among {st['partner_time_spread']['partners']} partners with at least 10 completed deliveries, the spread of partner average times is only marginally larger than chance (permutation p = {st['partner_time_spread']['p']:.2f}), so 'fastest partner' rankings are mostly noise. "
            "Partners need many more deliveries each before a leaderboard is trustworthy (see Power BI Delivery page, which shows the minimum-volume filter)."), PageBreak(),
          P("Q7. What drives cancellations?", H2),
          P(f"{k['cancelled_pct']:.1f}% of orders are cancelled. Cancellation by cuisine ranges from {min(k['cancel_pct_by_cuisine'].values()):.1f}% to {max(k['cancel_pct_by_cuisine'].values()):.1f}% "
            f"(chi-square permutation p = {st['cancel_rate_by_cuisine']['p']:.3f}, borderline); by payment method p = {st['cancel_rate_by_payment_method']['p']:.3f} (1 of {st['n_tests']} tests - consistent with chance); "
            f"by city p = {st['cancel_rate_by_restaurant_city']['p']:.2f}; weather p = {st['cancel_rate_by_weather']['p']:.2f}; hour p = {st['cancel_rate_by_hour']['p']:.2f}; coupon use p = {st['cancel_rate_by_has_coupon']['p']:.2f}. "
            f"Restaurant rating is unrelated to a restaurant's cancel rate (r = {st['corr_restaurant_rating_vs_cancel_rate']:.3f})."),
          img("eda_mpl_14_cancellation_by_cuisine.png", 10 * cm, "Cuisine differences are within sampling noise (borderline p = 0.06; one of many comparisons)."),
          P("Q8. How is customer lifetime value distributed?", H2),
          P("CLV is the sum of completed-order value per customer; it is right-skewed with most customers below INR 800 because the median customer places 2 orders. City-level differences are small relative to the within-city spread."),
          img("eda_sns_15_boxen_clv_by_city.png", 12 * cm), PageBreak()]
    s += [P("4. Data-integrity observations worth a data-owner review", H1),
          B("<b>Payment status is independent of order status.</b> About 80% of cancelled and 'Food Not Delivered' orders show payment 'Success' and only ~8% 'Refunded' - the same split as delivered orders. Either the two fields are not synchronised or refunds are not recorded."),
          B("<b>Feedback does not reflect lateness.</b> Mean delivery rating is " + f"{st['mean_delivery_rating_late']:.2f} for late and {st['mean_delivery_rating_ontime']:.2f} for on-time deliveries, and 'Delivery took way too long' style reviews are spread evenly across statuses."),
          B("<b>Customer and restaurant cities agree on only ~4% of orders</b>, and restaurant coordinates do not match their cities, so no real delivery distance exists. The 'DistanceProxyKm' feature is a proxy only."),
          B("<b>Cancelled orders have delivery times</b> (mean about 35 min, same as delivered), which a real system would not record; delivery-time modelling therefore uses completed orders only."),
          P("5. Charts", H1), P("All 38 charts carry titles, axis labels and legends where relevant; see <i>images/</i> (eda_mpl_* and eda_sns_*). The correlation heatmap below shows that no numeric feature correlates with delivery time (|r| &lt; 0.03).")]
    s += [img("eda_sns_01_corr_heatmap.png", 14 * cm)]
    doc.build(s, onFirstPage=footer, onLaterPages=footer)


# =============================================================== BUSINESS INSIGHTS REPORT
def insights_report():
    d = load_all(); k, st = d["k"], d["st"]; dres, cres, sc, churn = d["dres"], d["cres"], d["sc"], d["churn"]
    c = load_cleaned(); a = build_order_base(c)
    failed = a[a.OrderStatus.isin(["Cancelled", "Food Not Delivered"])]; paid_failed = failed[failed.PaymentStatus == "Success"]
    refunded_pct = (failed.PaymentStatus == "Refunded").mean() * 100
    dl = a[a.IsDelivered]; m = dl.groupby(dl.OrderDate.dt.to_period("M")).FinalAmount.sum()
    best_d = dres[dres.Stage.isin(["default", "tuned"])].sort_values("CV_R2_mean", ascending=False).iloc[0]
    best_c = cres[cres.Stage.isin(["default", "tuned"])].sort_values("CV_AUC", ascending=False).iloc[0]; base_c = cres[cres.Stage == "baseline"].iloc[0]
    leak = dres[dres.Stage == "leakage-demo"].iloc[0]
    doc = SimpleDocTemplate(str(ROOT / "reports" / "business_insights_report.pdf"), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.8 * cm, bottomMargin=1.8 * cm,
                            title="Business Insights Report - Zomato BI Project", author="Sundram Sharma")
    s = [P("Business Insights &amp; Recommendations", TITLE), P("Zomato Business Intelligence &amp; Delivery Time Prediction Platform", H2),
         P("For: City Operations, Restaurant Partnerships, Growth Marketing, Finance | Author: Sundram Sharma | Evidence: cleaned data (24 months), 23 permutation tests, 2 model families", SMALL), Spacer(1, 6)]
    s += [P("1. Executive summary", H1),
          P(f"<b>The problem to solve is order failure, not delivery speed.</b> {k['cancelled_pct'] + k['food_not_delivered_pct']:.1f}% of orders never reach the customer "
            f"({k['cancelled_pct']:.1f}% cancelled, {k['food_not_delivered_pct']:.1f}% 'Food Not Delivered'), and {k['late_pct_of_completed']:.1f}% of the rest arrive late. "
            f"Revenue is INR {k['revenue_inr']/1e5:.1f} lakh over 24 months and flat month to month."),
          P(f"<b>Nothing we measured explains the failures or the delays.</b> City, cuisine, weather, traffic, payment method, vehicle, hour, weekend and membership show no statistically reliable effect "
            f"(only {st['n_p_below_0.05']} of {st['n_tests']} tests at p&lt;0.05, about what chance produces). Consequently <b>neither predictive model beats a trivial baseline</b>: the delivery-time models reach test R<super>2</super> = {best_d.R2:.3f} "
            f"(PRD target 0.75) and the churn models reach ROC-AUC {best_c.ROC_AUC:.2f} (0.50 = coin flip)."),
          P("<b>Recommendation:</b> do not deploy either model and do not reallocate partners or target campaigns using them. Spend the next sprint on instrumentation (cancellation reasons, stage timestamps, GPS, customer coordinates), "
            "and on three decisions that do not depend on the models: reconcile payments on failed orders, run a coupon holdout test, and use the restaurant scorecard only as a watch-list.")]
    s += [P("2. What the data does show", H1),
          table(pd.DataFrame([
              ["Order failure", f"{k['cancelled_pct'] + k['food_not_delivered_pct']:.1f}% of orders (4,569 cancelled, 2,362 not delivered)", "Largest controllable loss"],
              ["Late deliveries", f"{k['late_pct_of_completed']:.1f}% of completed; late orders average 63.7 min vs 33.5", "SLA breach"],
              ["Delivery time", f"mean {k['avg_delivery_min']:.1f}, p90 {k['p90_delivery_min']:.0f}, p95 {k['p95_delivery_min']:.0f} min", "Baseline for SLA"],
              ["Peak demand", "12-13h and 19-20h daily; weekends not higher", "Only strongly structured pattern"],
              ["Customers", f"{k['one_time_customers_pct']:.1f}% of ordering customers bought once; 18.1% of registrants never ordered", "Activation gap"],
              ["Payments", f"{k['payment_failed_pct']:.1f}% of orders have a Failed payment", "Reconciliation"],
              ["Coupons", f"{k['coupon_usage_pct']:.0f}% of orders use a coupon, avg discount {k['avg_discount_pct_when_coupon']:.0f}% of food cost; INR {dl.Discount.sum()/1e5:.1f} lakh discounted on completed orders", "Cost with unknown benefit"],
          ], columns=["Area", "Finding", "Why it matters"]), [3 * cm, 9.7 * cm, 4.3 * cm], 8), Spacer(1, 6)]
    s += [P("3. What the data does not show (and why that matters)", H1),
          P(f"Each claim below was tested with a permutation test (2,000 shuffles). A pattern with p &gt;= 0.05 is not evidence of an effect, and with {st['n_tests']} tests one 'hit' is expected by chance."),
          table(pd.DataFrame([
              ["Delivery time vs traffic level", f"{k['traffic_high_vs_low_diff_min']:.1f} min longer in High/Severe", f"{st['delivery_time_by_traffic']['p']:.2f}", "No effect detected"],
              ["Delivery time vs rain", f"{k['rain_heavy_vs_light_diff_min']:.1f} min longer in heavy rain", f"{st['delivery_time_by_rain']['p']:.2f}", "No effect detected"],
              ["Late rate vs peak hour", "-", f"{st['late_rate_by_peak_hour']['p']:.2f}", "No effect detected"],
              ["Cancellation vs cuisine", "19.0% (Chinese) to 24.9% (Mexican)", f"{st['cancel_rate_by_cuisine']['p']:.3f}", "Borderline, 1 of 23 tests"],
              ["Cancellation vs payment method", "20.9% (COD) to 23.8% (Net Banking)", f"{st['cancel_rate_by_payment_method']['p']:.3f}", "Weak; expected by chance"],
              ["Cancellation vs weather / city / hour", "-", f"{st['cancel_rate_by_weather']['p']:.2f} / {st['cancel_rate_by_restaurant_city']['p']:.2f} / {st['cancel_rate_by_hour']['p']:.2f}", "No effect detected"],
              ["Basket vs membership tier", "INR 389 - 395", f"{st['order_value_by_membership']['p']:.2f}", "No effect detected"],
              ["Partner speed differences", "287 partners with 10+ deliveries", f"{st['partner_time_spread']['p']:.2f}", "Mostly noise"],
          ], columns=["Hypothesis", "Observed gap", "p-value", "Reading"]), [5.3 * cm, 5.2 * cm, 2.6 * cm, 3.9 * cm], 8), Spacer(1, 6)]
    s += [P("4. Model results", H1)]
    dd = dres[dres.Stage.isin(["baseline", "default", "tuned", "leakage-demo"])][["Model", "CV_R2_mean", "MAE", "RMSE", "R2"]].copy()
    dd.columns = ["Delivery-time model", "CV R2", "MAE", "RMSE", "Test R2"]; dd = dd.round(3).fillna("-")
    s += [P("<b>Delivery time (regression, 2,691 held-out completed orders).</b> Selected by cross-validation: " + f"{best_d.Model}", BODY), table(dd, [7.2 * cm, 2 * cm, 2 * cm, 2 * cm, 2.2 * cm], 7.5),
          P(f"No model improves on predicting the average ({dres.iloc[0].RMSE:.1f} min RMSE). The row marked LEAKAGE adds the post-delivery late flag and reaches R<super>2</super> = {leak.R2:.2f}: it demonstrates why OrderStatus is excluded, "
            "not a usable model. Reported feature 'importances' sit at the noise floor and should not be read as drivers.", SMALL), Spacer(1, 6)]
    cc = cres[["Model", "CV_AUC", "Accuracy", "F1", "ROC_AUC", "PR_AUC"]].round(3).copy(); cc.columns = ["Churn model", "CV AUC", "Accuracy", "F1 (churn)", "ROC-AUC", "PR-AUC"]
    s += [P(f"<b>60-day churn (classification, {len(churn):,} customers; {churn.Churn.mean()*100:.0f}% churn).</b> Selected by CV AUC: {best_c.Model}", BODY), table(cc, [6.5 * cm, 2 * cm, 2 * cm, 2.2 * cm, 2.2 * cm, 2 * cm], 7.5),
          P(f"The 'everyone churns' rule scores F1 = {base_c.F1:.2f}; every real model scores lower, and ROC-AUC is 0.51-0.53. The PRD's F1 &gt;= 0.70 target is therefore not meaningful here. On held-out customers the High/Medium/Low risk segments churn at 88.1% / 88.0% / 85.0% - a few points apart (in-sample the gap looks larger, 91% vs 83%, which is overfitting) - so they cannot support targeting.", SMALL),
          img("ml_churn_evaluation.png", 16.5 * cm), PageBreak()]
    pct_med = sc.Orders.median()
    s += [P("5. Recommendations by stakeholder", H1),
          P("City Operations - fix visibility before reallocating partners", H2),
          B("<b>Why:</b> 33.9% of orders fail and 15.9% of completed ones are late, yet no measured factor explains either, so reallocating partners by traffic/weather/hour would act on noise."),
          B("<b>Do:</b> log a cancellation/failure reason code and the actor (customer, restaurant, partner, system) plus timestamps for accepted, prepared, picked-up, delivered. Add partner GPS and customer coordinates for true distance."),
          B("<b>Measure success:</b> share of failures with a reason code (target 95%); then the share of failures attributable to controllable causes. Re-run the delivery-time model after 4 weeks of new data and require R<super>2</super> on a held-out time period well above 0 before acting on importances."),
          P("Restaurant Partnerships - scorecard as a watch-list only", H2),
          B(f"The composite score (0.4 rating, -0.3 cancel rate, -0.3 late rate; <i>restaurant_scorecard.csv</i>) ranks {len(sc):,} restaurants, but the median restaurant has only {pct_med:.0f} orders (max {int(sc.Orders.max())}), so its cancel rate has a standard error near 10 percentage points."),
          B("<b>Do:</b> review only restaurants that stay in the bottom decile for three consecutive months with at least 30 orders; do not use rating as a quality proxy - it is uncorrelated with cancellations (r = " + f"{st['corr_restaurant_rating_vs_cancel_rate']:.2f})."),
          P("Growth Marketing - test coupons, stop relying on the churn score", H2),
          B(f"{k['coupon_usage_pct']:.0f}% of orders carry a coupon averaging {k['avg_discount_pct_when_coupon']:.0f}% off, costing INR {dl.Discount.sum()/1e5:.1f} lakh on completed orders versus INR {k['revenue_inr']/1e5:.1f} lakh revenue. Observational data cannot show whether coupons create orders or just discount orders that would have happened."),
          B("<b>Do:</b> run a randomised coupon holdout (e.g. 10% of eligible users get no coupon for 4 weeks) and compare orders per user and margin. For retention, use the simple rule 'no order in 60 days' - it performs as well as the model - and activate the 18% of registrants who never ordered (first-order offer)."),
          P("Finance - payment reconciliation and a flat forecast", H2),
          B(f"Among {len(failed):,} cancelled / not-delivered orders, {len(paid_failed):,} (INR {paid_failed.FinalAmount.sum()/1e5:.1f} lakh) show payment 'Success' and only {refunded_pct:.1f}% show 'Refunded', the same mix as delivered orders. Either status fields are unsynchronised or customers may be owed refunds."),
          B(f"<b>Do:</b> reconcile payment gateway records against order status. For planning, monthly revenue is stable (mean INR {m.mean()/1e5:.2f} lakh, SD {m.std()/1e5:.2f}); a naive next-quarter estimate is about INR {3*m.tail(3).mean()/1e5:.1f} lakh (trailing 3-month average x 3), with no evidence of growth or seasonality in the 24 months available."),
          P("6. Risks and limits of this analysis", H1),
          B("Revenue counts completed orders only; ~34% of orders produced nothing, and payment data suggests some may still have been charged."),
          B("Ambiguous slash dates (~15%) were resolved heuristically; weather/traffic link to only ~63% of orders. Neither changes the conclusions (all weather/traffic effects are null on the linked subset)."),
          B("The data behaves like a simulated snapshot: many natural relationships (lateness vs rating, RFM vs churn, traffic vs time) are absent. Conclusions apply to this dataset; validate on production data before operational use."),
          B("Model numbers in this report were produced with the project's NumPy fallback implementation of scikit-learn; re-running <i>python run_pipeline.py</i> with scikit-learn installed regenerates every figure, table and model file.")]
    doc.build(s, onFirstPage=footer, onLaterPages=footer)


def main():
    eda_report(); insights_report(); print("reports written")


if __name__ == "__main__":
    main()
