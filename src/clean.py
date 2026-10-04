"""clean.py - Table-by-table cleaning for the Zomato BI project.

The DataCleaner class keeps state (a transformation log) while each
clean_<table> method applies documented rules.  Every decision is recorded in
`self.log` and exported to data/cleaned/cleaning_log.csv.

Key evidence-based decisions (verified on the raw data, see notebook 01):
  * Negative FoodCost / DeliveryTimeMinutes / Price / AverageCost / Age / etc.
    are SIGN FLIPS, not true negatives: |FoodCost| + DeliveryFee - Discount + GST
    reproduces FinalAmount exactly for all 181 negative rows. We therefore take
    the absolute value instead of discarding the rows.
  * Slash dates mix DD/MM/YYYY and MM/DD/YYYY inside the SAME column. Values
    with a part > 12 are unambiguous; ambiguous ones default to DD/MM (Indian
    convention) except orders, where we pick the candidate closest to the
    payment date of the same order.
  * Weather/traffic rows are stored in contiguous per-city blocks, so a missing
    City can be recovered from its neighbours.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from .ingest import load_raw, CLEAN_DIR

LOG = logging.getLogger("clean")

# Historical / alternate city names -> canonical names used in cities.csv
CITY_ALIASES = {
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "bombay": "Mumbai",
    "calcutta": "Kolkata", "baroda": "Vadodara", "new delhi": "Delhi",
    "vizag": "Visakhapatnam", "cochin": "Kochi", "poona": "Pune",
    "madras": "Chennai", "gurgaon": "Delhi",
}


# ----------------------------------------------------------------------------
# generic helpers
# ----------------------------------------------------------------------------
def parse_mixed_dates(s: pd.Series, default_dayfirst: bool = True):
    """Parse a string Series that mixes ISO, YYYY.MM.DD, DD-MM-YYYY, 'DD Mon YYYY'
    and slash formats whose day/month order is itself mixed.

    Returns (parsed, ambiguous, alt):
      parsed    - datetime64 Series using the best guess
      ambiguous - bool Series, True where a slash date has both parts <= 12
      alt       - the alternative interpretation for ambiguous rows (else NaT)
    """
    s = s.astype("string").str.strip()
    out = pd.Series(pd.NaT, index=s.index, dtype="datetime64[ns]")
    alt = out.copy()

    iso = s.str.match(r"^\d{4}-\d{2}-\d{2}$", na=False)
    out[iso] = pd.to_datetime(s[iso], format="%Y-%m-%d", errors="coerce")
    dot = s.str.match(r"^\d{4}\.\d{2}\.\d{2}$", na=False)
    out[dot] = pd.to_datetime(s[dot], format="%Y.%m.%d", errors="coerce")
    dash = s.str.match(r"^\d{2}-\d{2}-\d{4}$", na=False)      # always DD-MM-YYYY (checked)
    out[dash] = pd.to_datetime(s[dash], format="%d-%m-%Y", errors="coerce")
    mon = s.str.match(r"^\d{1,2} [A-Za-z]{3} \d{4}$", na=False)
    out[mon] = pd.to_datetime(s[mon], format="%d %b %Y", errors="coerce")

    sl = s.str.match(r"^\d{2}/\d{2}/\d{4}$", na=False)
    a = pd.to_numeric(s.str.slice(0, 2), errors="coerce")
    b = pd.to_numeric(s.str.slice(3, 5), errors="coerce")
    dmy = pd.to_datetime(s.where(sl), format="%d/%m/%Y", errors="coerce")
    mdy = pd.to_datetime(s.where(sl), format="%m/%d/%Y", errors="coerce")
    out[sl & (a > 12)] = dmy[sl & (a > 12)]
    out[sl & (b > 12)] = mdy[sl & (b > 12)]
    amb = sl & (a <= 12) & (b <= 12)
    out[amb] = (dmy if default_dayfirst else mdy)[amb]
    alt[amb] = (mdy if default_dayfirst else dmy)[amb]
    return out, amb.fillna(False), alt


def clean_text(s: pd.Series) -> pd.Series:
    """Strip whitespace, collapse internal spaces; blanks -> NaN."""
    s = s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
    return s.mask(s.isin(["", "nan", "NaN", "None"]))


class DataCleaner:
    """Cleans all 12 tables and records every transformation."""

    def __init__(self, raw: dict | None = None):
        self.raw = raw if raw is not None else load_raw()
        self.clean: dict = {}
        self.log: list = []
        self.cities: list = []

    # ------------------------------------------------------------------ utils
    def _log(self, table, issue, column, count, action):
        count = int(count)
        if count:
            self.log.append(dict(table=table, issue=issue, column=column,
                                 rows_affected=count, action=action))
            LOG.info("%-18s %-28s %-20s %6d  %s", table, issue, column, count, action)

    def _dedupe(self, df, table, subset=None, keep="first"):
        n = df.duplicated(subset=subset, keep=keep).sum()
        self._log(table, "duplicate rows", ",".join(subset) if subset else "all", n,
                  f"dropped duplicates (keep={keep})")
        return df.drop_duplicates(subset=subset, keep=keep)

    def _to_num(self, df, cols):
        for c in cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        return df

    def _abs(self, df, table, col):
        n = (df[col] < 0).sum()
        self._log(table, "negative value (sign flip)", col, n, "abs() - verified sign error")
        df[col] = df[col].abs()
        return df

    def _canon_city(self, s: pd.Series, table: str, col: str = "City") -> pd.Series:
        """Normalise whitespace/case, map aliases, fuzzy-match to canonical list."""
        import difflib
        raw = clean_text(s)
        key = raw.str.lower()
        mapped = key.map(CITY_ALIASES)
        title = raw.str.title()
        canon_lc = {c.lower(): c for c in self.cities}
        out = key.map(canon_lc).fillna(mapped)
        unresolved = out.isna() & raw.notna()
        for v in key[unresolved].unique():
            m = difflib.get_close_matches(v, list(canon_lc), n=1, cutoff=0.8)
            if m:
                out[key == v] = canon_lc[m[0]]
        changed = (out.fillna("") != s.astype("string").fillna("")).sum()
        self._log(table, "city spelling/case/whitespace", col, changed, "canonicalised to cities.City")
        return out.astype("string")

    # ----------------------------------------------------------------- tables
    def clean_cities(self):
        d = self.raw["cities"].copy()
        d = self._to_num(d, ["CityID", "Population", "AverageIncome"])
        d["City"] = clean_text(d["City"]).str.title()
        n = d["AverageIncome"].isna().sum()
        d["AverageIncome"] = d["AverageIncome"].fillna(d.groupby("Region")["AverageIncome"].transform("median"))
        self._log("cities", "missing", "AverageIncome", n, "imputed with region median")
        self.cities = d["City"].tolist()
        self.clean["cities"] = d

    def clean_customers(self):
        d = self.raw["customers"].copy()
        d = self._dedupe(d, "customers")
        d = self._dedupe(d, "customers", subset=["CustomerID"])
        for c in ["Name", "Email", "Gender", "State", "Membership", "PreferredCuisine"]:
            d[c] = clean_text(d[c])
        d["City"] = self._canon_city(d["City"], "customers")
        d = self._to_num(d, ["CustomerID", "Age", "TotalOrders"])
        # Age: sign flips -> abs; implausible (>100) -> NaN; missing -> median
        d = self._abs(d, "customers", "Age")
        bad = (d["Age"] > 100) | (d["Age"] < 10)
        self._log("customers", "implausible", "Age", bad.sum(), "set NaN then imputed")
        d.loc[bad, "Age"] = np.nan
        n = d["Age"].isna().sum()
        d["Age"] = d["Age"].fillna(d["Age"].median()).round().astype(int)
        self._log("customers", "missing", "Age", n, "imputed with median")
        # Phone: exactly 10 digits, else NaN
        ph = d["Phone"].astype("string").str.replace(r"\.0$", "", regex=True).str.strip()
        valid = ph.str.fullmatch(r"\d{10}").fillna(False)
        self._log("customers", "invalid/missing phone", "Phone", (~valid).sum(), "set to NaN (not guessable)")
        d["Phone"] = ph.where(valid)
        # Pincode: 6 digits (strip sign) else NaN
        pc = d["Pincode"].astype("string").str.replace(r"\.0$", "", regex=True).str.lstrip("-").str.strip()
        valid = pc.str.fullmatch(r"\d{6}").fillna(False)
        self._log("customers", "malformed pincode", "Pincode", (~valid).sum(), "set to NaN")
        d["Pincode"] = pc.where(valid)
        # categorical NaN -> 'Unknown' (kept explicit rather than guessing)
        for c in ["Gender", "Membership", "PreferredCuisine", "State"]:
            self._log("customers", "missing", c, d[c].isna().sum(), "filled 'Unknown'")
            d[c] = d[c].fillna("Unknown")
        self._log("customers", "missing", "Email", d["Email"].isna().sum(), "left NULL (nullable)")
        # dates
        dt, amb, _ = parse_mixed_dates(d["RegistrationDate"])
        self._log("customers", "mixed date formats", "RegistrationDate", len(d), "parsed to ISO")
        self._log("customers", "ambiguous slash date", "RegistrationDate", amb.sum(), "defaulted DD/MM")
        d["RegistrationDate"] = dt
        d["TotalOrders"] = d["TotalOrders"].fillna(0).clip(lower=0).astype(int)
        self.clean["customers"] = d

    def clean_restaurants(self):
        d = self.raw["restaurants"].copy()
        d = self._dedupe(d, "restaurants", subset=["RestaurantID"])
        for c in ["RestaurantName", "Cuisine", "Area", "OwnerName", "RestaurantType"]:
            d[c] = clean_text(d[c])
        d["City"] = self._canon_city(d["City"], "restaurants")
        d = self._to_num(d, ["RestaurantID", "Rating", "AverageCost", "Latitude", "Longitude"])
        self._log("restaurants", "missing", "Cuisine", d["Cuisine"].isna().sum(), "filled 'Unknown'")
        d["Cuisine"] = d["Cuisine"].fillna("Unknown")
        d = self._abs(d, "restaurants", "AverageCost")
        bad = d["Rating"] > 5
        self._log("restaurants", "rating out of 1-5", "Rating", bad.sum(), "set NaN then imputed")
        d.loc[bad, "Rating"] = np.nan
        n = d["Rating"].isna().sum()
        d["Rating"] = d["Rating"].fillna(d.groupby("Cuisine")["Rating"].transform("median")).fillna(d["Rating"].median()).round(1)
        self._log("restaurants", "missing", "Rating", n, "imputed with cuisine median")
        self.clean["restaurants"] = d

    def clean_menu(self):
        d = self.raw["menu"].copy()
        d = self._dedupe(d, "menu", subset=["FoodItemID"])
        d = self._to_num(d, ["FoodItemID", "RestaurantID", "Price", "PreparationTime", "Calories"])
        d["FoodName"] = clean_text(d["FoodName"]); d["Category"] = clean_text(d["Category"])
        d = self._abs(d, "menu", "Price")
        n = d["Calories"].isna().sum()
        d["Calories"] = d["Calories"].fillna(d.groupby("Category")["Calories"].transform("median")).round()
        self._log("menu", "missing", "Calories", n, "imputed with category median")
        before = len(d)
        d = d[d["RestaurantID"].isin(self.clean["restaurants"]["RestaurantID"])]
        self._log("menu", "orphan FK", "RestaurantID", before - len(d), "dropped")
        self.clean["menu"] = d

    def clean_delivery_partners(self):
        d = self.raw["delivery_partners"].copy()
        d = self._dedupe(d, "delivery_partners", subset=["DeliveryPartnerID"])
        d = self._to_num(d, ["DeliveryPartnerID", "Age", "Rating", "CompletedDeliveries", "AverageDeliveryTime"])
        d["Name"] = clean_text(d["Name"]); d["VehicleType"] = clean_text(d["VehicleType"])
        d["City"] = self._canon_city(d["City"], "delivery_partners")
        d = self._abs(d, "delivery_partners", "AverageDeliveryTime")
        n = d["Rating"].isna().sum()
        d["Rating"] = d["Rating"].fillna(d["Rating"].median())
        self._log("delivery_partners", "missing", "Rating", n, "imputed with median")
        dt, amb, _ = parse_mixed_dates(d["JoiningDate"])
        self._log("delivery_partners", "mixed date formats", "JoiningDate", len(d), "parsed to ISO")
        d["JoiningDate"] = dt
        # IQR outlier flag (vectorised) - flagged, not removed
        q1, q3 = np.percentile(d["AverageDeliveryTime"], [25, 75]); iqr = q3 - q1
        out = (d["AverageDeliveryTime"] > q3 + 3 * iqr) | (d["AverageDeliveryTime"] < q1 - 3 * iqr)
        self._log("delivery_partners", "extreme outlier (IQR*3)", "AverageDeliveryTime", out.sum(), "capped at bound")
        d["AverageDeliveryTime"] = d["AverageDeliveryTime"].clip(q1 - 3 * iqr, q3 + 3 * iqr)
        self.clean["delivery_partners"] = d

    def clean_promotions(self):
        d = self.raw["promotions"].copy()
        d["CouponCode"] = clean_text(d["CouponCode"]).str.upper()
        d["CampaignName"] = clean_text(d["CampaignName"])
        d = self._to_num(d, ["PromotionID", "DiscountPercentage"])
        sd, _, _ = parse_mixed_dates(d["StartDate"]); ed, _, _ = parse_mixed_dates(d["EndDate"])
        sw = ed < sd
        self._log("promotions", "EndDate before StartDate", "StartDate/EndDate", sw.sum(), "swapped")
        d["StartDate"] = sd.where(~sw, ed); d["EndDate"] = ed.where(~sw, sd)
        d = self._dedupe(d, "promotions", subset=["CouponCode"])
        self.clean["promotions"] = d

    def clean_weather(self):
        d = self.raw["weather"].copy()
        d = self._to_num(d, ["WeatherID", "Temperature", "Rainfall", "Humidity"])
        d = d.sort_values("WeatherID").reset_index(drop=True)
        # recover missing City from the contiguous per-city block
        city = self._canon_city(d["City"], "weather")
        n = city.isna().sum()
        city = city.where(~(city.isna() & (city.ffill() == city.bfill())), city.ffill())
        self._log("weather", "missing", "City", n - city.isna().sum(), "recovered from neighbouring rows (per-city blocks)")
        d["City"] = city.ffill()
        dt, amb, _ = parse_mixed_dates(d["Date"])
        self._log("weather", "mixed date formats", "Date", len(d), "parsed to ISO")
        self._log("weather", "ambiguous slash date", "Date", amb.sum(), "defaulted DD/MM (unresolvable: dates are sampled with repeats)")
        d["Date"] = dt
        d["WeatherCondition"] = clean_text(d["WeatherCondition"])
        d = self._abs(d, "weather", "Humidity"); d["Humidity"] = d["Humidity"].clip(upper=100)
        hi = np.percentile(d["Rainfall"].dropna(), 99)
        n = d["Rainfall"].isna().sum()
        d["Rainfall"] = d["Rainfall"].fillna(d.groupby("WeatherCondition")["Rainfall"].transform("median"))
        self._log("weather", "missing", "Rainfall", n, "imputed with weather-condition median")
        self._log("weather", "extreme outlier", "Rainfall", (d["Rainfall"] > hi).sum(), f"capped at P99={hi:.0f}mm")
        d["Rainfall"] = d["Rainfall"].clip(upper=hi)
        t_hi = np.percentile(d["Temperature"], 99.5)
        self._log("weather", "extreme outlier", "Temperature", (d["Temperature"] > t_hi).sum(), f"capped at P99.5={t_hi:.1f}C")
        d["Temperature"] = d["Temperature"].clip(upper=t_hi)
        # NOTE: (City, Date) is NOT unique - dates are sampled with repeats, so rows are kept and
        # aggregated (mean) when joined to orders (see features.py).
        self.clean["weather"] = d

    def clean_traffic(self):
        d = self.raw["traffic"].copy()
        d = self._to_num(d, ["TrafficID", "AverageSpeed"])
        d = d.sort_values("TrafficID").reset_index(drop=True)
        d["City"] = self._canon_city(d["City"], "traffic")
        dt, amb, _ = parse_mixed_dates(d["Date"])
        self._log("traffic", "mixed date formats", "Date", len(d), "parsed to ISO")
        self._log("traffic", "ambiguous slash date", "Date", amb.sum(), "defaulted DD/MM (unresolvable: dates are sampled with repeats)")
        d["Date"] = dt
        d = self._abs(d, "traffic", "AverageSpeed")
        hi = np.percentile(d["AverageSpeed"], 99)
        self._log("traffic", "extreme outlier", "AverageSpeed", (d["AverageSpeed"] > hi).sum(), f"capped at P99={hi:.0f}km/h")
        d["AverageSpeed"] = d["AverageSpeed"].clip(upper=hi)
        # TrafficLevel from AverageSpeed (class medians -> nearest class)
        d["TrafficLevel"] = clean_text(d["TrafficLevel"])
        med = d.groupby("TrafficLevel")["AverageSpeed"].median()
        miss = d["TrafficLevel"].isna()
        nearest = d.loc[miss, "AverageSpeed"].apply(lambda v: (med - v).abs().idxmin())
        d.loc[miss, "TrafficLevel"] = nearest
        self._log("traffic", "missing", "TrafficLevel", miss.sum(), "inferred from AverageSpeed (nearest class median)")
        d["Time"] = d["Time"].astype("string").str.slice(0, 5)
        self.clean["traffic"] = d

    def clean_orders(self):
        d = self.raw["orders"].copy()
        d = self._dedupe(d, "orders")
        d = self._dedupe(d, "orders", subset=["OrderID"])
        d = self._to_num(d, ["OrderID", "CustomerID", "RestaurantID", "DeliveryPartnerID", "DeliveryTimeMinutes",
                             "FoodCost", "DeliveryFee", "Discount", "GST", "FinalAmount"])
        for c in ["OrderStatus", "PaymentMethod"]:
            d[c] = clean_text(d[c])
        d["CouponCode"] = clean_text(d["CouponCode"]).str.upper()
        # dates - resolve ambiguous slash dates with payment date of same order
        dt, amb, alt = parse_mixed_dates(d["OrderDate"])
        pay_dt, pamb, _ = parse_mixed_dates(self.raw["payments"].drop_duplicates("OrderID")["PaymentDate"])
        ref = pd.Series(pay_dt.where(~pamb).values,
                        index=pd.to_numeric(self.raw["payments"].drop_duplicates("OrderID")["OrderID"]))
        r = d["OrderID"].map(ref)
        use_alt = amb & r.notna() & alt.notna() & ((alt - r).abs() < (dt - r).abs())
        self._log("orders", "ambiguous slash date", "OrderDate", amb.sum(), "resolved via payment date, else DD/MM")
        self._log("orders", "ambiguous date flipped", "OrderDate", use_alt.sum(), "MM/DD chosen (closer to payment date)")
        dt = dt.where(~use_alt, alt)
        d["OrderDate"] = dt
        d["OrderTime"] = d["OrderTime"].astype("string").str.slice(0, 8)
        # numeric repairs
        for c in ["FoodCost", "DeliveryTimeMinutes"]:
            d = self._abs(d, "orders", c)
        zero = d["DeliveryTimeMinutes"] <= 0
        self._log("orders", "zero delivery time", "DeliveryTimeMinutes", zero.sum(), "set NaN")
        d.loc[zero, "DeliveryTimeMinutes"] = np.nan
        out = d["DeliveryTimeMinutes"] > 180
        self._log("orders", "unrealistic delivery time (>180 min)", "DeliveryTimeMinutes", out.sum(), "set NaN (excluded from modelling)")
        d.loc[out, "DeliveryTimeMinutes"] = np.nan
        calc = d["FoodCost"] + d["DeliveryFee"] - d["Discount"] + d["GST"]
        n = d["FinalAmount"].isna().sum()
        d["FinalAmount"] = d["FinalAmount"].fillna(calc)
        self._log("orders", "missing", "FinalAmount", n, "recomputed = FoodCost+Fee-Discount+GST (formula holds on 100% of valid rows)")
        d["DeliveryFee"] = d["DeliveryFee"].abs()
        # orphan foreign keys
        keep = (d["CustomerID"].isin(self.clean["customers"]["CustomerID"]) &
                d["RestaurantID"].isin(self.clean["restaurants"]["RestaurantID"]) &
                d["DeliveryPartnerID"].isin(self.clean["delivery_partners"]["DeliveryPartnerID"]))
        self._log("orders", "orphan FK (customer/restaurant/partner)", "CustomerID,RestaurantID", (~keep).sum(),
                  "dropped (cannot be attributed)")
        d = d[keep]
        bad = d["CouponCode"].notna() & ~d["CouponCode"].isin(self.clean["promotions"]["CouponCode"])
        self._log("orders", "unknown coupon", "CouponCode", bad.sum(), "set NULL")
        d.loc[bad, "CouponCode"] = np.nan
        self.clean["orders"] = d

    def clean_order_items(self):
        d = self.raw["order_items"].copy()
        d = self._dedupe(d, "order_items", subset=["OrderItemID"])
        d = self._to_num(d, ["OrderItemID", "OrderID", "FoodItemID", "Quantity", "UnitPrice", "TotalPrice"])
        d = self._abs(d, "order_items", "Quantity")
        n = d["TotalPrice"].isna().sum()
        d["TotalPrice"] = d["TotalPrice"].fillna(d["Quantity"] * d["UnitPrice"])
        self._log("order_items", "missing", "TotalPrice", n, "recomputed Quantity*UnitPrice")
        keep = d["OrderID"].isin(self.clean["orders"]["OrderID"]) & d["FoodItemID"].isin(self.clean["menu"]["FoodItemID"])
        self._log("order_items", "orphan FK", "OrderID,FoodItemID", (~keep).sum(), "dropped")
        self.clean["order_items"] = d[keep]

    def clean_payments(self):
        d = self.raw["payments"].copy()
        d = self._dedupe(d, "payments")
        d = self._to_num(d, ["PaymentID", "OrderID"])
        for c in ["PaymentMethod", "PaymentStatus", "TransactionID"]:
            d[c] = clean_text(d[c])
        dt, amb, _ = parse_mixed_dates(d["PaymentDate"])
        self._log("payments", "mixed date formats", "PaymentDate", len(d), "parsed to ISO")
        d["PaymentDate"] = dt
        # one payment per order: prefer Success, then the latest PaymentID
        rank = d["PaymentStatus"].map({"Success": 0, "Refunded": 1, "Pending": 2, "Failed": 3}).fillna(4)
        d = d.assign(_r=rank).sort_values(["OrderID", "_r", "PaymentID"], ascending=[True, True, False])
        n = d.duplicated("OrderID").sum()
        self._log("payments", "multiple payments per order", "OrderID", n, "kept best-status (Success first) row")
        d = d.drop_duplicates("OrderID").drop(columns="_r").sort_values("PaymentID")
        self._log("payments", "missing", "TransactionID", d["TransactionID"].isna().sum(), "left NULL")
        keep = d["OrderID"].isin(self.clean["orders"]["OrderID"])
        self._log("payments", "orphan FK", "OrderID", (~keep).sum(), "dropped")
        self.clean["payments"] = d[keep]

    def clean_feedback(self):
        d = self.raw["customer_feedback"].copy()
        d = self._dedupe(d, "customer_feedback")
        d = self._dedupe(d, "customer_feedback", subset=["OrderID"])
        d = self._to_num(d, ["FeedbackID", "OrderID", "CustomerRating", "DeliveryRating", "FoodRating"])
        for c in ["DeliveryRating", "FoodRating", "CustomerRating"]:
            bad = d[c] > 5
            self._log("customer_feedback", "rating out of 1-5", c, bad.sum(), "set NaN then imputed")
            d.loc[bad, c] = np.nan
        for c, others in [("CustomerRating", ["DeliveryRating", "FoodRating"]),
                          ("DeliveryRating", ["CustomerRating", "FoodRating"]),
                          ("FoodRating", ["CustomerRating", "DeliveryRating"])]:
            n = d[c].isna().sum()
            d[c] = d[c].fillna(d[others].mean(axis=1)).fillna(d[c].median()).round()
            self._log("customer_feedback", "missing", c, n, "imputed with mean of other two ratings")
        d["Review"] = clean_text(d["Review"])
        self._log("customer_feedback", "missing", "Review", d["Review"].isna().sum(), "left NULL")
        d["Sentiment"] = clean_text(d["Sentiment"])
        keep = d["OrderID"].isin(self.clean["orders"]["OrderID"])
        self._log("customer_feedback", "orphan FK", "OrderID", (~keep).sum(), "dropped")
        self.clean["customer_feedback"] = d[keep].astype({c: int for c in ["CustomerRating", "DeliveryRating", "FoodRating"]})

    # -------------------------------------------------------------------- run
    def run(self) -> dict:
        """Run all steps in FK dependency order. Raises on unexpected failures."""
        steps = [self.clean_cities, self.clean_customers, self.clean_restaurants, self.clean_menu,
                 self.clean_delivery_partners, self.clean_promotions, self.clean_weather, self.clean_traffic,
                 self.clean_orders, self.clean_order_items, self.clean_payments, self.clean_feedback]
        for step in steps:
            try:
                step()
            except Exception:
                LOG.exception("cleaning step failed: %s", step.__name__)
                raise
        return self.clean

    def validate(self) -> pd.DataFrame:
        """Post-clean integrity checks (PK uniqueness, FK coverage, nulls in key columns)."""
        c = self.clean
        checks = []
        pk = {"cities": "CityID", "customers": "CustomerID", "restaurants": "RestaurantID", "menu": "FoodItemID",
              "delivery_partners": "DeliveryPartnerID", "promotions": "PromotionID", "orders": "OrderID",
              "order_items": "OrderItemID", "payments": "PaymentID", "customer_feedback": "FeedbackID",
              "weather": "WeatherID", "traffic": "TrafficID"}
        for t, k in pk.items():
            checks.append((f"{t}.{k} unique", bool(c[t][k].is_unique)))
        fks = [("customers", "City", "cities", "City"), ("restaurants", "City", "cities", "City"),
               ("delivery_partners", "City", "cities", "City"), ("weather", "City", "cities", "City"),
               ("traffic", "City", "cities", "City"), ("menu", "RestaurantID", "restaurants", "RestaurantID"),
               ("orders", "CustomerID", "customers", "CustomerID"), ("orders", "RestaurantID", "restaurants", "RestaurantID"),
               ("orders", "DeliveryPartnerID", "delivery_partners", "DeliveryPartnerID"),
               ("order_items", "OrderID", "orders", "OrderID"), ("order_items", "FoodItemID", "menu", "FoodItemID"),
               ("payments", "OrderID", "orders", "OrderID"), ("customer_feedback", "OrderID", "orders", "OrderID")]
        for ct, cc, pt, pc in fks:
            checks.append((f"FK {ct}.{cc} -> {pt}.{pc}", bool(c[ct][cc].dropna().isin(c[pt][pc]).all())))
        o = c["orders"]
        checks += [("orders no negative FoodCost", bool((o.FoodCost >= 0).all())),
                   ("orders FinalAmount no nulls", bool(o.FinalAmount.notna().all())),
                   ("orders DeliveryTime in (0,180]", bool(o.DeliveryTimeMinutes.dropna().between(0, 180).all())),
                   ("restaurants Rating in 1-5", bool(c["restaurants"].Rating.between(1, 5).all())),
                   ("customers Age in 10-100", bool(c["customers"].Age.between(10, 100).all())),
                   ("promotions End>=Start", bool((c["promotions"].EndDate >= c["promotions"].StartDate).all()))]
        return pd.DataFrame(checks, columns=["check", "passed"])

    # integer columns must be written without a ".0" suffix or PostgreSQL \copy into SMALLINT/INTEGER fails
    INT_COLS = {
        "cities": ["CityID", "Population"], "customers": ["CustomerID", "Age", "TotalOrders"],
        "restaurants": ["RestaurantID"], "menu": ["FoodItemID", "RestaurantID", "PreparationTime", "Calories"],
        "delivery_partners": ["DeliveryPartnerID", "Age", "CompletedDeliveries"], "promotions": ["PromotionID", "DiscountPercentage"],
        "orders": ["OrderID", "CustomerID", "RestaurantID", "DeliveryPartnerID", "DeliveryTimeMinutes"],
        "order_items": ["OrderItemID", "OrderID", "FoodItemID", "Quantity"], "payments": ["PaymentID", "OrderID"],
        "customer_feedback": ["FeedbackID", "OrderID", "CustomerRating", "DeliveryRating", "FoodRating"],
        "weather": ["WeatherID", "Humidity"], "traffic": ["TrafficID"]}

    def save(self, directory: Path = CLEAN_DIR):
        """Write cleaned CSVs + cleaning_log.csv + validation_report.csv."""
        directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
        for t, df in self.clean.items():
            out = df.copy()
            for col in self.INT_COLS.get(t, []):
                out[col] = pd.to_numeric(out[col], errors="coerce").round().astype("Int64")
            out.to_csv(directory / f"{t}.csv", index=False)
        pd.DataFrame(self.log).to_csv(directory / "cleaning_log.csv", index=False)
        self.validate().to_csv(directory / "validation_report.csv", index=False)


def main():
    """CLI: python -m src.clean  (from project root)"""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    cl = DataCleaner(); cl.run(); cl.save()
    v = cl.validate()
    print(f"\nValidation: {int(v.passed.sum())}/{len(v)} checks passed")
    print(f"Cleaning log entries: {len(cl.log)}; rows touched: {sum(x['rows_affected'] for x in cl.log)}")


if __name__ == "__main__":
    main()
