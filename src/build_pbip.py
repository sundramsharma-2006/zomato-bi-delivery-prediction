"""build_pbip.py - Generate a Power BI Project (PBIP) for the 6-page dashboard.

Output (powerbi/):
    zomato_dashboard.pbip                       <- double-click to open in Power BI Desktop
    zomato_dashboard.SemanticModel/             TMDL model: 17 tables (CSV import), 11 relationships, 38 DAX measures
    zomato_dashboard.Report/                    PBIR report: 6 pages, ~55 visuals, Zomato theme

File formats follow Microsoft's published PBIR / TMDL / PBIP schemas (visualContainer 2.0.0, page 2.0.0,
report 3.0.0, version 2.0.0). The project was generated WITHOUT access to Power BI Desktop, so it has been
syntax-checked but not opened; see powerbi/README_BUILD.md for what to do if Desktop reports an error.

Run: python -m src.build_pbip     (after `python -m src.export_powerbi`)
"""
import hashlib
import json
import re
import shutil
from pathlib import Path

import pandas as pd

from .ingest import ROOT

PB = ROOT / "powerbi"
DATA = PB / "data"
NAME = "zomato_dashboard"
SM = PB / f"{NAME}.SemanticModel"
RP = PB / f"{NAME}.Report"
SCHEMA = "https://developer.microsoft.com/json-schemas/fabric/item/report/definition"

# ------------------------------------------------------------------ semantic model
TABLES = ["dim_date", "dim_city", "dim_customer", "dim_restaurant", "dim_partner", "dim_promotion", "dim_menu",
          "fact_orders", "fact_order_items", "rpt_restaurant_top10", "rpt_restaurant_bottom10",
          "ml_delivery_predictions", "ml_delivery_feature_importance", "ml_delivery_model_comparison",
          "ml_churn_scores", "ml_churn_feature_importance", "ml_churn_model_comparison"]
TEXT_FORCE = {"Phone", "Pincode", "TransactionID"}
DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# (name, DAX, format, folder)
MEASURES = [
    ("Total Orders", "COUNTROWS ( fact_orders )", "#,0", "KPIs"),
    ("Completed Orders", "CALCULATE ( COUNTROWS ( fact_orders ), fact_orders[IsDelivered] = 1 )", "#,0", "KPIs"),
    ("Total Revenue", "CALCULATE ( SUM ( fact_orders[FinalAmount] ), fact_orders[IsDelivered] = 1 )", "#,0", "KPIs"),
    ("Avg Order Value", "DIVIDE ( [Total Revenue], [Completed Orders] )", "#,0", "KPIs"),
    ("Avg Delivery Time", "CALCULATE ( AVERAGE ( fact_orders[DeliveryTimeMinutes] ), fact_orders[IsDelivered] = 1 )", "0.0", "KPIs"),
    ("P90 Delivery Time", "PERCENTILEX.INC ( FILTER ( fact_orders, fact_orders[IsDelivered] = 1 && NOT ISBLANK ( fact_orders[DeliveryTimeMinutes] ) ), fact_orders[DeliveryTimeMinutes], 0.9 )", "0", "KPIs"),
    ("Avg Rating", "AVERAGE ( fact_orders[AvgRating] )", "0.00", "KPIs"),
    ("Late %", "DIVIDE ( CALCULATE ( COUNTROWS ( fact_orders ), fact_orders[IsLate] = 1 ), [Completed Orders] )", "0.0%", "KPIs"),
    ("Cancellation %", "DIVIDE ( CALCULATE ( COUNTROWS ( fact_orders ), fact_orders[IsCancelled] = 1 ), [Total Orders] )", "0.0%", "KPIs"),
    ("Failure %", "DIVIDE ( CALCULATE ( COUNTROWS ( fact_orders ), fact_orders[IsDelivered] = 0 ), [Total Orders] )", "0.0%", "KPIs"),
    ("Active Customers", "VAR EndD = MAX ( dim_date[Date] ) RETURN CALCULATE ( DISTINCTCOUNT ( fact_orders[CustomerID] ), DATESBETWEEN ( dim_date[Date], EndD - 59, EndD ) )", "#,0", "KPIs"),
    ("Revenue PM", "CALCULATE ( [Total Revenue], DATEADD ( dim_date[Date], -1, MONTH ) )", "#,0", "Time intelligence"),
    ("Revenue MoM %", "DIVIDE ( [Total Revenue] - [Revenue PM], [Revenue PM] )", "0.0%", "Time intelligence"),
    ("Customers", "COUNTROWS ( dim_customer )", "#,0", "Customers"),
    ("Ordering Customers", "CALCULATE ( COUNTROWS ( dim_customer ), dim_customer[Orders] >= 1 )", "#,0", "Customers"),
    ("Repeat Customers", "CALCULATE ( COUNTROWS ( dim_customer ), dim_customer[Orders] >= 2 )", "#,0", "Customers"),
    ("Retention Rate", "DIVIDE ( [Repeat Customers], [Ordering Customers] )", "0.0%", "Customers"),
    ("One-time Customers %", "DIVIDE ( CALCULATE ( COUNTROWS ( dim_customer ), dim_customer[CustomerType] = \"One-time\" ), [Ordering Customers] )", "0.0%", "Customers"),
    ("Avg CLV", "CALCULATE ( AVERAGE ( dim_customer[CLV] ), dim_customer[Orders] >= 1 )", "#,0", "Customers"),
    ("New Customers", "CALCULATE ( COUNTROWS ( dim_customer ), USERELATIONSHIP ( dim_customer[RegistrationDate], dim_date[Date] ) )", "#,0", "Customers"),
    ("Avg Composite Score", "AVERAGE ( dim_restaurant[CompositeScore] )", "0.00", "Restaurants"),
    ("Avg Restaurant Rating", "AVERAGE ( dim_restaurant[Rating] )", "0.00", "Restaurants"),
    ("Top 10 Score", "SUM ( rpt_restaurant_top10[CompositeScore] )", "0.00", "Restaurants"),
    ("Bottom 10 Score", "SUM ( rpt_restaurant_bottom10[CompositeScore] )", "0.00", "Restaurants"),
    ("Partner Deliveries", "[Completed Orders]", "#,0", "Delivery"),
    ("Avg Time (10+ deliveries)", "IF ( [Completed Orders] >= 10, [Avg Delivery Time] )", "0.0", "Delivery"),
    ("Total Discount", "CALCULATE ( SUM ( fact_orders[Discount] ), fact_orders[IsDelivered] = 1 )", "#,0", "Sales"),
    ("Coupon Usage %", "DIVIDE ( SUM ( fact_orders[HasCoupon] ), [Total Orders] )", "0.0%", "Sales"),
    ("Discount % of Revenue", "DIVIDE ( [Total Discount], [Total Revenue] + [Total Discount] )", "0.0%", "Sales"),
    ("Delivery MAE", "AVERAGEX ( ml_delivery_predictions, ABS ( ml_delivery_predictions[Residual] ) )", "0.00", "ML"),
    ("Delivery RMSE", "SQRT ( AVERAGEX ( ml_delivery_predictions, ml_delivery_predictions[Residual] ^ 2 ) )", "0.00", "ML"),
    ("Delivery R2", "1 - DIVIDE ( SUMX ( ml_delivery_predictions, ml_delivery_predictions[Residual] ^ 2 ), SUMX ( ml_delivery_predictions, ( ml_delivery_predictions[Actual] - AVERAGE ( ml_delivery_predictions[Actual] ) ) ^ 2 ) )", "0.000", "ML"),
    ("Avg Predicted", "AVERAGE ( ml_delivery_predictions[Predicted] )", "0.0", "ML"),
    ("Avg Actual", "AVERAGE ( ml_delivery_predictions[Actual] )", "0.0", "ML"),
    ("Delivery Importance", "SUM ( ml_delivery_feature_importance[permutation_mse_increase] )", "0.000", "ML"),
    ("Churn Importance", "SUM ( ml_churn_feature_importance[auc_drop_when_shuffled] )", "0.0000", "ML"),
    ("Test Customers", "CALCULATE ( COUNTROWS ( ml_churn_scores ), ml_churn_scores[Split] = \"test\" )", "#,0", "ML"),
    ("Actual Churn % (test)", "CALCULATE ( AVERAGE ( ml_churn_scores[Churn] ), ml_churn_scores[Split] = \"test\" )", "0.0%", "ML"),
]

RELATIONSHIPS = [  # (name, from_table.col, to_table.col, active)
    ("rel_orders_date", "fact_orders.OrderDate", "dim_date.Date", True),
    ("rel_orders_customer", "fact_orders.CustomerID", "dim_customer.CustomerID", True),
    ("rel_orders_restaurant", "fact_orders.RestaurantID", "dim_restaurant.RestaurantID", True),
    ("rel_orders_partner", "fact_orders.DeliveryPartnerID", "dim_partner.DeliveryPartnerID", True),
    ("rel_orders_city", "fact_orders.City", "dim_city.City", True),
    ("rel_orders_promo", "fact_orders.CouponCode", "dim_promotion.CouponCode", True),
    ("rel_items_orders", "fact_order_items.OrderID", "fact_orders.OrderID", True),
    ("rel_items_menu", "fact_order_items.FoodItemID", "dim_menu.FoodItemID", True),
    ("rel_customer_regdate", "dim_customer.RegistrationDate", "dim_date.Date", False),
    ("rel_pred_orders", "ml_delivery_predictions.OrderID", "fact_orders.OrderID", True),
    ("rel_churn_customer", "ml_churn_scores.CustomerID", "dim_customer.CustomerID", True),
]
SORT_BY = {("dim_date", "MonthName"): "MonthNum", ("dim_date", "DayOfWeek"): "DayOfWeekNum", ("dim_customer", "CLVBand"): "CLVBandOrder"}


def col_types(table):
    """Infer (column, kind) with kind in text/int/num/date from the exported CSV."""
    df = pd.read_csv(DATA / f"{table}.csv", nrows=5000)
    out = []
    for c in df.columns:
        s = df[c].dropna()
        if c in TEXT_FORCE: kind = "text"
        elif len(s) and s.astype(str).map(lambda v: bool(DATE_RE.match(v))).all(): kind = "date"
        elif pd.api.types.is_integer_dtype(df[c]): kind = "int"
        elif pd.api.types.is_float_dtype(df[c]): kind = "num"
        else: kind = "text"
        out.append((c, kind))
    return out


def q(name):  # TMDL name quoting
    return f"'{name}'" if re.search(r"[^A-Za-z0-9_]", name) else name


def table_tmdl(table):
    T = "\t"
    cols = col_types(table)
    m_types = {"text": "type text", "int": "Int64.Type", "num": "type number", "date": "type date"}
    dt = {"text": "string", "int": "int64", "num": "double", "date": "dateTime"}
    lines = [f"table {table}"]
    if table == "dim_date": lines.append(f"{T}dataCategory: Time")
    lines.append("")
    for c, kind in cols:
        lines.append(f"{T}column {q(c)}")
        lines.append(f"{T*2}dataType: {dt[kind]}")
        if kind == "date": lines.append(f"{T*2}formatString: Long Date")
        if kind == "int": lines.append(f"{T*2}formatString: 0")
        if table == "dim_date" and c == "Date": lines.append(f"{T*2}isKey")
        lines.append(f"{T*2}summarizeBy: none")
        lines.append(f"{T*2}sourceColumn: {c}")
        if (table, c) in SORT_BY: lines.append(f"{T*2}sortByColumn: {SORT_BY[(table, c)]}")
        lines.append("")
    if table == "fact_orders":
        for name, dax, fmt, folder in MEASURES:
            lines.append(f"{T}measure {q(name)} = {dax}")
            lines.append(f"{T*2}formatString: {fmt}")
            lines.append(f"{T*2}displayFolder: {folder}")
            lines.append("")
    types = ", ".join(f'{{"{c}", {m_types[k]}}}' for c, k in cols)
    lines += [f"{T}partition {table} = m", f"{T*2}mode: import", f"{T*2}source =",
              f"{T*4}let",
              f'{T*5}Source = Csv.Document(File.Contents(DataPath & "\\{table}.csv"), [Delimiter=",", Encoding=65001, QuoteStyle=QuoteStyle.Csv]),',
              f"{T*5}Promoted = Table.PromoteHeaders(Source, [PromoteAllScalars=true]),",
              f'{T*5}Typed = Table.TransformColumnTypes(Promoted, {{{types}}}, "en-US")',
              f"{T*4}in", f"{T*5}Typed", ""]
    return "\n".join(lines)


def write(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def build_model():
    pass  # overwrite in place (stale files are removed by deleting the .SemanticModel/.Report folders before a rebuild)
    write(SM / "definition.pbism", json.dumps({"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/semanticModel/definitionProperties/1.0.0/schema.json",
                                              "version": "4.0", "settings": {}}, indent=2))
    d = SM / "definition"
    write(d / "database.tmdl", "database\n\tcompatibilityLevel: 1600\n")
    refs = "\n".join(f"ref table {t}" for t in TABLES)
    write(d / "model.tmdl", "model Model\n\tculture: en-US\n\tdefaultPowerBIDataSourceVersion: powerBI_V3\n\tdiscourageImplicitMeasures\n\n" + refs + "\n\nref culture en-US\n")
    write(d / "cultures" / "en-US.tmdl", "cultureInfo en-US\n")
    data_path = str(DATA).replace("/sessions/loving-awesome-archimedes/mnt/internship", r"C:\Users\hp\OneDrive\Desktop\internship").replace("/", "\\")
    write(d / "expressions.tmdl", f'expression DataPath = "{data_path}" meta [IsParameterQuery=true, Type="Text", IsParameterQueryRequired=true]\n')
    rel = []
    for n, f, t, act in RELATIONSHIPS:
        rel += [f"relationship {n}"] + ([] if act else ["\tisActive: false"]) + [f"\tfromColumn: {f}", f"\ttoColumn: {t}", ""]
    write(d / "relationships.tmdl", "\n".join(rel))
    for t in TABLES:
        write(d / "tables" / f"{t}.tmdl", table_tmdl(t))


# ------------------------------------------------------------------ report
def nid(key):
    return hashlib.md5(key.encode()).hexdigest()[:20]


def lit(v):
    return {"expr": {"Literal": {"Value": v}}}


def txt(s):
    return lit("'" + s.replace("'", "''") + "'")


def col(entity, prop):
    return {"kind": "Column", "entity": entity, "prop": prop}


def mea(entity, prop):
    return {"kind": "Measure", "entity": entity, "prop": prop}


def field_json(f):
    return {f["kind"]: {"Expression": {"SourceRef": {"Entity": f["entity"]}}, "Property": f["prop"]}}


def projection(f, active=False):
    p = {"field": field_json(f), "queryRef": f"{f['entity']}.{f['prop']}", "nativeQueryRef": f["prop"]}
    if active: p["active"] = True
    return p


class Page:
    def __init__(self, title, key):
        self.title, self.key, self.visuals = title, key, []
        self.id = nid("page" + key)
        self.z = 0

    def add(self, vtype, x, y, w, h, roles=None, title=None, sort=None, objects=None, sync=None, name=None):
        self.z += 1000
        name = nid(f"{self.key}-{len(self.visuals)}-{vtype}")
        v = {"$schema": f"{SCHEMA}/visualContainer/2.0.0/schema.json", "name": name,
             "position": {"x": x, "y": y, "z": self.z, "height": h, "width": w, "tabOrder": self.z},
             "visual": {"visualType": vtype}}
        if roles:
            qs = {role: {"projections": [projection(f, active=(role == "Category" and f["kind"] == "Column")) for f in fields]} for role, fields in roles.items()}
            v["visual"]["query"] = {"queryState": qs}
            if sort: v["visual"]["query"]["sortDefinition"] = {"sort": [{"field": field_json(sort[0]), "direction": sort[1]}]}
        if objects: v["visual"]["objects"] = objects
        vco = {"border": [{"properties": {"show": lit("false")}}]}
        if title is not None:
            vco["title"] = [{"properties": {"show": lit("true"), "text": txt(title)}}]
        v["visual"]["visualContainerObjects"] = vco
        if sync: v["visual"]["syncGroup"] = {"groupName": sync, "fieldChanges": True, "filterChanges": True}
        v["visual"]["drillFilterOtherVisuals"] = True
        self.visuals.append((name, v))

    def textbox(self, runs, x, y, w, h):
        """runs: list of (text, font size pt)"""
        self.z += 1000
        name = nid(f"{self.key}-tb-{len(self.visuals)}")
        paras = [{"textRuns": [{"value": t, "textStyle": {"fontSize": f"{s}pt"}}]} for t, s in runs]
        v = {"$schema": f"{SCHEMA}/visualContainer/2.0.0/schema.json", "name": name,
             "position": {"x": x, "y": y, "z": self.z, "height": h, "width": w},
             "visual": {"visualType": "textbox", "objects": {"general": [{"properties": {"paragraphs": paras}}]},
                        "visualContainerObjects": {"title": [{"properties": {"show": lit("false")}}], "background": [{"properties": {"show": lit("false")}}],
                                                   "border": [{"properties": {"show": lit("false")}}]},
                        "drillFilterOtherVisuals": True}}
        self.visuals.append((name, v))

    def header(self, subtitle=None, slicers=True):
        self.textbox([(self.title, 22)], 20, 8, 820 if slicers else 1240, 38)
        if subtitle: self.textbox([(subtitle, 11)], 20, 46, 820 if slicers else 1240, 44)
        if slicers:
            self.add("slicer", 860, 8, 180, 52, {"Values": [col("dim_city", "City")]}, objects={"data": [{"properties": {"mode": txt("Dropdown")}}]}, sync="City")
            self.add("slicer", 1050, 8, 210, 52, {"Values": [col("dim_date", "Date")]}, objects={"data": [{"properties": {"mode": txt("Between")}}]}, sync="Date")

    def cards(self, measures, y=70, h=88):
        n = len(measures); w = (1240 - 10 * (n - 1)) / n
        for i, m in enumerate(measures):
            self.add("card", 20 + i * (w + 10), y, w, h, {"Values": [mea("fact_orders", m)]})


def build_pages():
    F = "fact_orders"
    pages = []
    # 1 Executive
    p = Page("1. Executive Dashboard", "exec"); p.header()
    p.cards(["Total Revenue", "Total Orders", "Avg Delivery Time", "Avg Rating", "Active Customers", "Late %", "Failure %", "P90 Delivery Time"])
    p.add("lineChart", 20, 175, 610, 250, {"Category": [col("dim_date", "MonthStart")], "Y": [mea(F, "Total Revenue")]}, title="Monthly revenue (INR)")
    p.add("clusteredBarChart", 650, 175, 610, 525, {"Category": [col("dim_city", "City")], "Y": [mea(F, "Total Revenue")]}, title="Revenue by city (INR)", sort=(mea(F, "Total Revenue"), "Descending"))
    p.add("donutChart", 20, 440, 610, 260, {"Category": [col(F, "OrderStatus")], "Y": [mea(F, "Total Orders")]}, title="Orders by status")
    pages.append(p)
    # 2 Customer
    p = Page("2. Customer Analytics", "cust"); p.header()
    p.cards(["Retention Rate", "One-time Customers %", "Avg CLV", "Ordering Customers", "Customers"])
    p.add("clusteredColumnChart", 20, 175, 400, 270, {"Category": [col("dim_customer", "Segment")], "Y": [mea(F, "Customers")]}, title="Customer segments (as of 31-Dec-2024)")
    p.add("donutChart", 430, 175, 400, 270, {"Category": [col("dim_customer", "CustomerType")], "Y": [mea(F, "Customers")]}, title="Repeat vs one-time")
    p.add("clusteredColumnChart", 840, 175, 420, 270, {"Category": [col("dim_customer", "CLVBand")], "Y": [mea(F, "Customers")]}, title="Customer lifetime value distribution (INR)")
    p.add("lineChart", 20, 460, 1240, 240, {"Category": [col("dim_date", "YearMonth")], "Y": [mea(F, "New Customers")]}, title="New registrations by month")
    pages.append(p)
    # 3 Restaurant
    p = Page("3. Restaurant Analytics", "rest"); p.header(subtitle="Composite score = 0.4 x rating - 0.3 x cancel rate - 0.3 x late rate (z-scores). Top/bottom lists use restaurants with 15+ orders; the median restaurant has only 17 orders, so treat as a watch-list.")
    p.cards(["Avg Composite Score", "Avg Restaurant Rating", "Total Revenue", "Cancellation %"], y=92, h=80)
    p.add("clusteredBarChart", 20, 185, 610, 255, {"Category": [col("rpt_restaurant_top10", "RestaurantLabel")], "Y": [mea(F, "Top 10 Score")]}, title="Top 10 restaurants by composite score", sort=(mea(F, "Top 10 Score"), "Descending"))
    p.add("clusteredBarChart", 650, 185, 610, 255, {"Category": [col("rpt_restaurant_bottom10", "RestaurantLabel")], "Y": [mea(F, "Bottom 10 Score")]}, title="Bottom 10 restaurants (watch-list)", sort=(mea(F, "Bottom 10 Score"), "Ascending"))
    p.add("clusteredBarChart", 20, 455, 610, 245, {"Category": [col("dim_restaurant", "Cuisine")], "Y": [mea(F, "Total Revenue")]}, title="Revenue by cuisine (INR)", sort=(mea(F, "Total Revenue"), "Descending"))
    p.add("clusteredBarChart", 650, 455, 610, 245, {"Category": [col("dim_restaurant", "Cuisine")], "Y": [mea(F, "Avg Restaurant Rating")]}, title="Average rating by cuisine")
    pages.append(p)
    # 4 Delivery
    p = Page("4. Delivery Analytics", "deliv"); p.header()
    p.cards(["Avg Delivery Time", "P90 Delivery Time", "Late %", "Completed Orders"])
    p.add("lineChart", 20, 175, 610, 250, {"Category": [col("dim_date", "MonthStart")], "Y": [mea(F, "Avg Delivery Time")]}, title="Average delivery time by month (min)")
    p.add("tableEx", 650, 175, 610, 525, {"Values": [col("dim_partner", "DeliveryPartnerID"), col("dim_partner", "VehicleType"), mea(F, "Partner Deliveries"), mea(F, "Avg Time (10+ deliveries)"), col("dim_partner", "Rating")]},
          title="Partner leaderboard (avg time shown only for 10+ deliveries)", sort=(mea(F, "Avg Time (10+ deliveries)"), "Ascending"))
    p.add("clusteredColumnChart", 20, 440, 300, 260, {"Category": [col(F, "TrafficScore")], "Y": [mea(F, "Avg Delivery Time")]}, title="By traffic (1 low - 4 severe)")
    p.add("clusteredColumnChart", 330, 440, 300, 260, {"Category": [col(F, "RainImpact")], "Y": [mea(F, "Avg Delivery Time")]}, title="By rain (0 none - 3 heavy)")
    pages.append(p)
    # 5 Sales
    p = Page("5. Sales Dashboard", "sales"); p.header()
    p.cards(["Total Revenue", "Avg Order Value", "Total Discount", "Coupon Usage %", "Discount % of Revenue"])
    p.add("lineChart", 20, 175, 810, 265, {"Category": [col("dim_date", "Date")], "Y": [mea(F, "Total Revenue")]}, title="Daily revenue (INR)")
    p.add("donutChart", 850, 175, 410, 265, {"Category": [col(F, "PaymentMethod")], "Y": [mea(F, "Total Orders")]}, title="Orders by payment method")
    p.add("clusteredBarChart", 20, 455, 610, 245, {"Category": [col("dim_promotion", "CampaignName")], "Y": [mea(F, "Total Discount")]}, title="Discount value by campaign (INR)", sort=(mea(F, "Total Discount"), "Descending"))
    p.add("clusteredColumnChart", 650, 455, 610, 245, {"Category": [col("dim_date", "YearMonth")], "Y": [mea(F, "Total Revenue")]}, title="Revenue by month (INR)")
    pages.append(p)
    # 6 ML
    p = Page("6. ML Dashboard", "ml"); p.header(subtitle="Model results are shown for transparency: neither model beats a naive baseline on this data (delivery R2 ~ 0.00, churn ROC-AUC ~ 0.53). See reports/business_insights_report.pdf. City/Date slicers do not apply to model outputs.", slicers=False)
    p.cards(["Delivery R2", "Delivery RMSE", "Delivery MAE", "Actual Churn % (test)"], y=92, h=80)
    p.add("scatterChart", 20, 185, 610, 290, {"Category": [col("ml_delivery_predictions", "OrderID")], "X": [mea(F, "Avg Predicted")], "Y": [mea(F, "Avg Actual")]}, title="Delivery time: predicted vs actual (test set)")
    p.add("clusteredBarChart", 650, 185, 610, 290, {"Category": [col("ml_delivery_feature_importance", "feature")], "Y": [mea(F, "Delivery Importance")]}, title="Delivery model: permutation importance (noise floor)", sort=(mea(F, "Delivery Importance"), "Descending"))
    p.add("clusteredColumnChart", 20, 490, 300, 210, {"Category": [col("ml_churn_scores", "RiskSegment")], "Y": [mea(F, "Test Customers")]}, title="Test customers by risk segment")
    p.add("clusteredColumnChart", 330, 490, 300, 210, {"Category": [col("ml_churn_scores", "RiskSegment")], "Y": [mea(F, "Actual Churn % (test)")]}, title="Actual churn % by segment (test)")
    p.add("clusteredBarChart", 650, 490, 610, 210, {"Category": [col("ml_churn_feature_importance", "feature")], "Y": [mea(F, "Churn Importance")]}, title="Churn model: top drivers (AUC drop)", sort=(mea(F, "Churn Importance"), "Descending"))
    pages.append(p)
    return pages


def build_report():
    pass
    write(RP / "definition.pbir", json.dumps({"$schema": "https://developer.microsoft.com/json-schemas/fabric/item/report/definitionProperties/2.0.0/schema.json",
                                              "version": "4.0", "datasetReference": {"byPath": {"path": f"../{NAME}.SemanticModel"}}}, indent=2))
    d = RP / "definition"
    write(d / "version.json", json.dumps({"$schema": f"{SCHEMA}/versionMetadata/1.0.0/schema.json", "version": "2.0.0"}, indent=2))
    theme_src = json.loads((PB / "zomato_theme.json").read_text(encoding="utf-8"))
    write(RP / "StaticResources" / "RegisteredResources" / "ZomatoTheme.json", json.dumps(theme_src, indent=2))
    report = {"$schema": f"{SCHEMA}/report/3.0.0/schema.json",
              "themeCollection": {"customTheme": {"name": "ZomatoTheme.json", "reportVersionAtImport": {"visual": "2.1.0", "report": "2.1.0", "page": "2.0.0"}, "type": "RegisteredResources"}},
              "resourcePackages": [{"name": "RegisteredResources", "type": "RegisteredResources", "items": [{"name": "ZomatoTheme.json", "path": "ZomatoTheme.json", "type": "CustomTheme"}]}],
              "settings": {"useStylableVisualContainerHeader": True, "exportDataMode": "AllowSummarized", "defaultDrillFilterOtherVisuals": True, "allowChangeFilterTypes": True, "useEnhancedTooltips": True}}
    write(d / "report.json", json.dumps(report, indent=2))
    pages = build_pages()
    write(d / "pages" / "pages.json", json.dumps({"$schema": f"{SCHEMA}/pagesMetadata/1.0.0/schema.json", "pageOrder": [p.id for p in pages], "activePageName": pages[0].id}, indent=2))
    n = 0
    for p in pages:
        write(d / "pages" / p.id / "page.json", json.dumps({"$schema": f"{SCHEMA}/page/2.0.0/schema.json", "name": p.id, "displayName": p.title, "displayOption": "FitToPage", "height": 720, "width": 1280}, indent=2))
        for name, v in p.visuals:
            write(d / "pages" / p.id / "visuals" / name / "visual.json", json.dumps(v, indent=2)); n += 1
    return len(pages), n


def main():
    build_model(); pages, visuals = build_report()
    write(PB / f"{NAME}.pbip", json.dumps({"$schema": "https://developer.microsoft.com/json-schemas/fabric/pbip/pbipProperties/1.0.0/schema.json", "version": "1.0",
                                            "artifacts": [{"report": {"path": f"{NAME}.Report"}}], "settings": {"enableAutoRecovery": True}}, indent=2))
    write(PB / ".gitignore", "**/.pbi/localSettings.json\n**/.pbi/cache.abf\n")
    print(f"PBIP written: {len(TABLES)} tables, {len(RELATIONSHIPS)} relationships, {len(MEASURES)} measures, {pages} pages, {visuals} visuals")


if __name__ == "__main__":
    main()
