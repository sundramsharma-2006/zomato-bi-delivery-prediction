"""model_delivery.py - Model 1: Delivery time regression (PRD 24.1).

Target : DeliveryTimeMinutes for COMPLETED deliveries (Delivered / Delivered Late).
         Cancelled / 'Food Not Delivered' orders have no meaningful delivery time.
Inputs : only information available when the order is placed/dispatched.
         OrderStatus, ratings and feedback are post-delivery -> excluded (leakage).
Flow   : 80/20 split -> 5-fold CV on train for 4 models -> randomized tuning of the
         two best (by CV) -> test metrics (MAE/MSE/RMSE/R2) -> importance + residuals.
Run    : python -m src.model_delivery
"""
import logging, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ingest import load_cleaned, ROOT
from .features import build_order_base
from . import ml_compat as ml

LOG = logging.getLogger("model_delivery")
SEED = 42
NUMERIC = ["TrafficScore", "AverageSpeed", "RainImpact", "Rainfall", "Temperature", "Humidity", "PeakHour",
           "WeekendOrder", "hour", "dow", "DistanceProxyKm", "DeliveryEfficiency", "PartnerRating",
           "CompletedDeliveries", "BasketSize", "ItemLines", "FoodCost", "DeliveryFee", "DiscountPct",
           "RestaurantRating", "AverageCost", "RestaurantPopularityRank", "CustomerTenure", "CustomerAge"]
CATEG = ["VehicleType", "WeatherCondition", "Membership", "RestaurantType"]


def make_matrix(abt: pd.DataFrame):
    """Return X (numeric+one-hot), y, and the modelling frame."""
    d = abt[abt["IsDelivered"] & abt["DeliveryTimeMinutes"].notna()].copy()
    X = pd.get_dummies(d[NUMERIC + CATEG], columns=CATEG, drop_first=False, dtype=float)
    X = X.fillna(X.median())
    return X, d["DeliveryTimeMinutes"].astype(float), d


def permutation_importance(model, X, y, n_repeats=5, seed=SEED):
    """Mean increase in MSE when a column is shuffled (vectorised, model-agnostic)."""
    rng = np.random.RandomState(seed); Xv = X.values.copy(); base = ml.mean_squared_error(y, model.predict(Xv)); out = []
    for j in range(Xv.shape[1]):
        inc = []
        for _ in range(n_repeats):
            col = Xv[:, j].copy(); Xv[:, j] = rng.permutation(col)
            inc.append(ml.mean_squared_error(y, model.predict(Xv)) - base); Xv[:, j] = col
        out.append(np.mean(inc))
    return pd.Series(out, index=X.columns)


def evaluate(model, X, y):
    p = model.predict(X)
    mse = ml.mean_squared_error(y, p)
    return dict(MAE=ml.mean_absolute_error(y, p), MSE=mse, RMSE=float(np.sqrt(mse)), R2=ml.r2_score(y, p))


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    img = ROOT / "images"; img.mkdir(exist_ok=True); (ROOT / "models").mkdir(exist_ok=True)
    c = load_cleaned(); abt = build_order_base(c)
    X, y, d = make_matrix(abt)
    Xtr, Xte, ytr, yte = ml.train_test_split(X, y, test_size=0.2, random_state=SEED)
    LOG.info("backend=%s  rows=%d  train=%d test=%d  features=%d", ml.BACKEND, len(X), len(Xtr), len(Xte), X.shape[1])

    # scaled copy for the linear model only
    sc = ml.StandardScaler().fit(Xtr)
    class Scaled:                                         # tiny wrapper so LinearRegression sees scaled data
        def __init__(s, m): s.m = m
        def fit(s, A, b): s.m.fit(sc.transform(A), b); return s
        def predict(s, A): return s.m.predict(sc.transform(A))
    models = {
        "Baseline (predict train mean)": None,
        "Linear Regression": Scaled(ml.LinearRegression()),
        "Decision Tree": ml.DecisionTreeRegressor(max_depth=6, min_samples_leaf=50, random_state=SEED),
        "Random Forest": ml.RandomForestRegressor(n_estimators=60, max_depth=8, min_samples_leaf=30, max_features=0.5, random_state=SEED),
        "Gradient Boosting": ml.GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, min_samples_leaf=30, random_state=SEED),
    }
    rows, fitted = [], {}
    base_pred = np.full(len(yte), ytr.mean())
    for name, m in models.items():
        t = time.time()
        if m is None:
            mse = ml.mean_squared_error(yte, base_pred)
            rows.append(dict(Model=name, CV_R2_mean=0.0, CV_R2_std=0.0, MAE=ml.mean_absolute_error(yte, base_pred), MSE=mse,
                             RMSE=np.sqrt(mse), R2=ml.r2_score(yte, base_pred), Stage="baseline")); continue
        if name == "Linear Regression":
            cv = ml.cross_val_score(ml.LinearRegression(), sc.transform(Xtr), ytr, cv=5, scoring="r2")
        else:
            cv = ml.cross_val_score(m, Xtr.values, ytr.values, cv=5, scoring="r2")
        m.fit(Xtr if name == "Linear Regression" else Xtr.values, ytr if name == "Linear Regression" else ytr.values)
        fitted[name] = m
        ev = evaluate(m, Xte if name == "Linear Regression" else Xte.values, yte)
        rows.append(dict(Model=name, CV_R2_mean=cv.mean(), CV_R2_std=cv.std(), **ev, Stage="default"))
        LOG.info("%-20s CV R2 %.4f  test R2 %.4f RMSE %.2f  (%.1fs)", name, cv.mean(), ev["R2"], ev["RMSE"], time.time() - t)

    # ---- hyper-parameter tuning of the two best TREE/ENSEMBLE models by CV R2 ----
    default = pd.DataFrame(rows); cand = default[default.Stage == "default"].sort_values("CV_R2_mean", ascending=False)
    grids = {
        "Random Forest": (ml.RandomForestRegressor(random_state=SEED), {"n_estimators": [40, 60], "max_depth": [4, 6, 8, 12], "min_samples_leaf": [20, 50, 100], "max_features": [0.3, 0.5]}),
        "Gradient Boosting": (ml.GradientBoostingRegressor(random_state=SEED), {"n_estimators": [50, 100, 200], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [2, 3, 4], "min_samples_leaf": [30, 80]}),
        "Decision Tree": (ml.DecisionTreeRegressor(random_state=SEED), {"max_depth": [3, 4, 6, 8], "min_samples_leaf": [30, 50, 100, 200]}),
        "Linear Regression": None,
    }
    tuned = {}
    tunable = cand[cand.Model.isin([k for k, v in grids.items() if v is not None])]   # Linear Regression has no hyper-parameters
    top2 = tunable.Model.head(2).tolist(); LOG.info("tuning the two best tunable models: %s", top2)
    for name in top2:
        if grids.get(name) is None: continue
        est, grid = grids[name]
        s = ml.RandomizedSearchCV(est, grid, n_iter=4, cv=3, scoring="r2", random_state=SEED).fit(Xtr.values, ytr.values)
        ev = evaluate(s.best_estimator_, Xte.values, yte); tuned[name + " (tuned)"] = s.best_estimator_
        rows.append(dict(Model=name + " (tuned)", CV_R2_mean=s.best_score_, CV_R2_std=np.nan, **ev, Stage="tuned", Params=str(s.best_params_)))
        LOG.info("%s tuned %s  CV R2 %.4f test R2 %.4f", name, s.best_params_, s.best_score_, ev["R2"])
    res = pd.DataFrame(rows)

    # ---- model selection on CROSS-VALIDATION (never on the test set) ----
    sel = res[res.Stage != "baseline"].sort_values("CV_R2_mean", ascending=False).iloc[0]
    best_name = sel.Model; best = tuned.get(best_name, fitted.get(best_name))
    LOG.info("selected by CV: %s", best_name)
    is_lin = best_name == "Linear Regression"
    Xte_m = Xte if is_lin else Xte.values
    pred = best.predict(Xte_m); resid = yte.values - pred

    # ---- importance (impurity for trees + permutation on test) ----
    perm = permutation_importance(best, Xte if is_lin else Xte, yte) if is_lin else permutation_importance(best, Xte, yte)
    imp = pd.DataFrame({"feature": X.columns, "permutation_mse_increase": perm.values})
    if hasattr(best, "feature_importances_"):
        imp["impurity_importance"] = best.feature_importances_
    imp = imp.sort_values("permutation_mse_increase", ascending=False)

    # ---- leakage demonstration (NOT a deployable model) ----
    Xl = X.copy(); Xl["LEAK_IsLate"] = d["IsLate"].values
    Xl_tr, Xl_te, yl_tr, yl_te = ml.train_test_split(Xl, y, test_size=0.2, random_state=SEED)
    leak = ml.GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, max_depth=3, min_samples_leaf=30, random_state=SEED).fit(Xl_tr.values, yl_tr.values)
    leak_ev = evaluate(leak, Xl_te.values, yl_te)
    rows.append(dict(Model="LEAKAGE DEMO: GB + OrderStatus flag (not usable)", CV_R2_mean=np.nan, CV_R2_std=np.nan, **leak_ev, Stage="leakage-demo"))
    res = pd.DataFrame(rows)

    # ---- artifacts ----
    out = ROOT / "data" / "cleaned"
    res.to_csv(out / "model_comparison_delivery.csv", index=False)
    imp.to_csv(out / "feature_importance_delivery.csv", index=False)
    pa = d.loc[Xte.index, ["OrderID", "RCity", "OrderDate"]].copy(); pa["Actual"] = yte.values; pa["Predicted"] = pred; pa["Residual"] = resid
    pa.to_csv(out / "delivery_predictions_test.csv", index=False)
    with open(ROOT / "models" / "delivery_time_best_model.pkl", "wb") as f:
        pickle.dump(dict(model=best, name=best_name, features=list(X.columns), scaler=sc if is_lin else None,
                         backend=ml.BACKEND, test_metrics=evaluate(best, Xte_m, yte)), f)

    # ---- plots ----
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    ax[0].scatter(pred, yte, s=6, alpha=.35, color="#E23744"); lo, hi = 0, max(yte.max(), pred.max())
    ax[0].plot([lo, hi], [lo, hi], "k--", label="perfect prediction"); ax[0].set(xlabel="Predicted minutes", ylabel="Actual minutes", title=f"Predicted vs actual - {best_name}")
    ax[0].legend(); ax[1].scatter(pred, resid, s=6, alpha=.35, color="#2D2D2D"); ax[1].axhline(0, color="#E23744")
    ax[1].set(xlabel="Predicted minutes", ylabel="Residual (actual - predicted)", title="Residuals vs predicted")
    plt.tight_layout(); plt.savefig(img / "ml_delivery_pred_vs_actual_residuals.png", dpi=130); plt.close()
    top = imp.head(12).iloc[::-1]; fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(top.feature, top.permutation_mse_increase, color="#E23744"); ax.set(xlabel="Increase in test MSE when shuffled", title=f"Permutation importance - {best_name}")
    plt.tight_layout(); plt.savefig(img / "ml_delivery_feature_importance.png", dpi=130); plt.close()
    fig, ax = plt.subplots(figsize=(8, 4.5)); r = res[res.Stage.isin(["baseline", "default", "tuned"])]
    ax.bar(r.Model, r.R2, color="#E23744"); ax.axhline(0.75, ls="--", color="k", label="PRD target R2 = 0.75"); ax.set(ylabel="Test R2", title="Delivery-time models: test R2"); ax.legend()
    plt.xticks(rotation=25, ha="right"); plt.tight_layout(); plt.savefig(img / "ml_delivery_model_comparison.png", dpi=130); plt.close()
    print(res.round(4).to_string(index=False)); print("\nTop features:\n", imp.head(8).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
