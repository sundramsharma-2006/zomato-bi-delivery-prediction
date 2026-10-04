"""stat_checks.py - Are the patterns we are about to report distinguishable from noise?

Permutation tests (NumPy only, 2,000 shuffles, seed 42). Output: reports/stat_checks.json
Rule used in the insights report: do not recommend action on a pattern with p >= 0.05
(and note multiple comparisons: ~1 in 20 tests would pass by chance).
Run: python -m src.stat_checks
"""
import json
import numpy as np
import pandas as pd
from .ingest import load_cleaned, ROOT
from .features import build_order_base

rng = np.random.RandomState(42)


def _codes(cat):
    return pd.factorize(pd.Series(cat).astype(str))[0]


def _chi2(codes, flag, k):
    ct = np.bincount(codes * 2 + flag, minlength=2 * k).reshape(k, 2).astype(float)
    exp = ct.sum(1, keepdims=True) * ct.sum(0, keepdims=True) / ct.sum()
    return ((ct - exp) ** 2 / np.where(exp > 0, exp, 1)).sum()


def perm_p_cat(cat, flag, n=2000):
    codes = _codes(cat); flag = np.asarray(flag).astype(int); k = codes.max() + 1; obs = _chi2(codes, flag, k)
    null = np.array([_chi2(codes, rng.permutation(flag), k) for _ in range(n)])
    return float(obs), float((null >= obs).mean() + 1 / (n + 1))


def perm_p_group_mean(cat, y, n=2000):
    """Between-group variance of group means vs permutation null."""
    y = np.asarray(y, float); ok = ~np.isnan(y); codes = _codes(np.asarray(cat)[ok]); y = y[ok]; k = codes.max() + 1; cnt = np.bincount(codes, minlength=k)
    f = lambda yy: np.var(np.bincount(codes, weights=yy, minlength=k) / cnt, ddof=1)
    obs = f(y); null = np.array([f(rng.permutation(y)) for _ in range(n)])
    return float(obs), float((null >= obs).mean() + 1 / (n + 1))


def main():
    c = load_cleaned(); a = build_order_base(c); comp = a[a.IsDelivered & a.DeliveryTimeMinutes.notna()]
    R = {}
    for name, cat in [("cuisine", a.Cuisine), ("restaurant_city", a.RCity), ("weather", a.WeatherCondition), ("payment_method", a.PaymentMethod),
                      ("hour", a.hour), ("weekend", a.WeekendOrder), ("membership", a.Membership), ("has_coupon", a.HasCoupon)]:
        s, p = perm_p_cat(cat, a.IsCancelled); R[f"cancel_rate_by_{name}"] = dict(chi2=s, p=p)
    for name, cat in [("restaurant_city", comp.RCity), ("traffic", comp.TrafficScore), ("rain", comp.RainImpact), ("vehicle", comp.VehicleType), ("peak_hour", comp.PeakHour), ("weekend", comp.WeekendOrder)]:
        s, p = perm_p_cat(cat, comp.IsLate); R[f"late_rate_by_{name}"] = dict(chi2=s, p=p)
    for name, cat in [("traffic", comp.TrafficScore), ("rain", comp.RainImpact), ("weather", comp.WeatherCondition), ("vehicle", comp.VehicleType), ("city", comp.RCity)]:
        s, p = perm_p_group_mean(cat, comp.DeliveryTimeMinutes); R[f"delivery_time_by_{name}"] = dict(stat=s, p=p)
    dl = a[a.IsDelivered]
    for name, cat in [("membership", dl.Membership), ("has_coupon", dl.HasCoupon), ("cuisine", dl.Cuisine)]:
        s, p = perm_p_group_mean(cat, dl.FinalAmount); R[f"order_value_by_{name}"] = dict(stat=s, p=p)
    R["coupon_aov_with"] = float(dl[dl.HasCoupon == 1].FinalAmount.mean()); R["coupon_aov_without"] = float(dl[dl.HasCoupon == 0].FinalAmount.mean())
    # delivery partners: is partner-to-partner spread larger than chance?
    pc = comp.groupby("DeliveryPartnerID").DeliveryTimeMinutes.agg(["mean", "count"]); pc = pc[pc["count"] >= 10]
    obs = pc["mean"].var(); ids = comp.DeliveryPartnerID.values; y = comp.DeliveryTimeMinutes.values; keep = np.isin(ids, pc.index)
    pc_codes = pd.factorize(ids[keep])[0]; kk = pc_codes.max() + 1; cnt = np.bincount(pc_codes, minlength=kk); nulls = []
    for _ in range(500):
        yy = rng.permutation(y)[keep]; nulls.append(np.var(np.bincount(pc_codes, weights=yy, minlength=kk) / cnt, ddof=1))
    R["partner_time_spread"] = dict(partners=int(len(pc)), observed_var=float(obs), p=float((np.array(nulls) >= obs).mean() + 1 / 501))
    # restaurant rating vs cancel rate
    rs = a.groupby("RestaurantID").agg(r=("RestaurantRating", "first"), cr=("IsCancelled", "mean"), n=("OrderID", "count")); rs = rs[rs.n >= 10]
    R["corr_restaurant_rating_vs_cancel_rate"] = float(rs.r.corr(rs.cr))
    # feedback rating vs lateness (post-delivery association; should exist if data is coherent)
    fb = comp.dropna(subset=["DeliveryRating"]); R["mean_delivery_rating_late"] = float(fb[fb.IsLate == 1].DeliveryRating.mean()); R["mean_delivery_rating_ontime"] = float(fb[fb.IsLate == 0].DeliveryRating.mean())
    R["delivered_late_mean_time"] = float(comp[comp.IsLate == 1].DeliveryTimeMinutes.mean()); R["delivered_mean_time"] = float(comp[comp.IsLate == 0].DeliveryTimeMinutes.mean())
    R["n_tests"] = int(sum(1 for k, v in R.items() if isinstance(v, dict) and "p" in v)); R["n_p_below_0.05"] = int(sum(1 for v in R.values() if isinstance(v, dict) and v.get("p", 1) < 0.05))
    with open(ROOT / "reports" / "stat_checks.json", "w") as fh: json.dump(R, fh, indent=2)
    for k, v in R.items(): print(k, v if not isinstance(v, dict) else {x: round(y, 4) for x, y in v.items()})


if __name__ == "__main__":
    main()
