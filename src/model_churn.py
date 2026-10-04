"""model_churn.py - Model 2: 60-day customer churn classification (PRD 24.2).

Label  : churn = 1 if the customer places NO order in the 60 days after the cutoff
         (cutoff = last order date - 60 days). Features use ONLY orders <= cutoff.
Imbalance: ~87% of customers churn (orders per customer are sparse), so we use
         class_weight='balanced' (documented choice) and report ROC-AUC / PR-AUC and
         the F1 of a trivial "everyone churns" baseline next to the model F1.
Run    : python -m src.model_churn
"""
import logging, pickle, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .ingest import load_cleaned, ROOT
from .features import build_order_base, build_churn_table
from . import ml_compat as ml

LOG = logging.getLogger("model_churn")
SEED = 42
NUMERIC = ["Recency", "Frequency", "Monetary", "AvgOrderValue", "CancelRate", "LateRate", "CouponRate", "AvgDeliveryTime",
           "AvgRating", "DistinctRestaurants", "WeekendShare", "FirstOrderAgeDays", "AvgGapDays", "Tenure", "Age"]


def metrics(y, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return dict(Accuracy=ml.accuracy_score(y, pred), Precision=ml.precision_score(y, pred), Recall=ml.recall_score(y, pred),
                F1=ml.f1_score(y, pred), F1_retained=ml.f1_score(1 - y, 1 - pred), ROC_AUC=ml.roc_auc_score(y, proba),
                PR_AUC=ml.average_precision_score(y, proba))


def permutation_auc(model, X, y, scaler=None, n_repeats=5, seed=SEED):
    rng = np.random.RandomState(seed); Xv = X.values.copy()
    f = (lambda A: model.predict_proba(scaler.transform(A))[:, 1]) if scaler else (lambda A: model.predict_proba(A)[:, 1])
    base = ml.roc_auc_score(y, f(Xv)); out = []
    for j in range(Xv.shape[1]):
        d = []
        for _ in range(n_repeats):
            col = Xv[:, j].copy(); Xv[:, j] = rng.permutation(col); d.append(base - ml.roc_auc_score(y, f(Xv))); Xv[:, j] = col
        out.append(np.mean(d))
    return pd.Series(out, index=X.columns)


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    img = ROOT / "images"; img.mkdir(exist_ok=True)
    c = load_cleaned(); abt = build_order_base(c); ch, cutoff = build_churn_table(abt, c)
    X = pd.get_dummies(ch[NUMERIC + ["Membership"]], columns=["Membership"], dtype=float); y = ch["Churn"].values
    idx = np.arange(len(X)); itr, ite = ml.train_test_split(idx, test_size=0.2, random_state=SEED, stratify=y)
    Xtr, Xte, ytr, yte = X.iloc[itr], X.iloc[ite], y[itr], y[ite]
    LOG.info("backend=%s cutoff=%s customers=%d churn_rate=%.3f", ml.BACKEND, cutoff.date(), len(X), y.mean())
    sc = ml.StandardScaler().fit(Xtr)

    class Scaled:
        def __init__(s, m): s.m = m
        def fit(s, A, b): s.m.fit(sc.transform(A), b); return s
        def predict_proba(s, A): return s.m.predict_proba(sc.transform(A))
        def predict(s, A): return s.m.predict(sc.transform(A))
    models = {
        "Logistic Regression": Scaled(ml.LogisticRegression(C=1.0, class_weight="balanced", max_iter=200)),
        "Decision Tree": ml.DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, class_weight="balanced", random_state=SEED),
        "Random Forest": ml.RandomForestClassifier(n_estimators=60, max_depth=8, min_samples_leaf=30, class_weight="balanced", random_state=SEED),
        "Gradient Boosting": ml.GradientBoostingClassifier(n_estimators=100, learning_rate=0.05, max_depth=3, min_samples_leaf=30, class_weight="balanced", random_state=SEED),
    }
    ones = np.ones(len(yte)); rows = [dict(Model="Baseline: predict everyone churns", CV_AUC=0.5, Stage="baseline", **metrics(yte, ones * 1.0))]
    rows[0]["ROC_AUC"] = 0.5; rows[0]["PR_AUC"] = yte.mean()
    fitted, probas = {}, {}
    for name, m in models.items():
        t = time.time()
        if name == "Logistic Regression":
            cv = ml.cross_val_score(ml.LogisticRegression(C=1.0, class_weight="balanced", max_iter=200), sc.transform(Xtr), ytr, cv=5, scoring="roc_auc")
            m.fit(Xtr.values, ytr)
        else:
            cv = ml.cross_val_score(m, Xtr.values, ytr, cv=5, scoring="roc_auc"); m.fit(Xtr.values, ytr)
        fitted[name] = m; pr = m.predict_proba(Xte.values)[:, 1]; probas[name] = pr
        rows.append(dict(Model=name, CV_AUC=cv.mean(), Stage="default", **metrics(yte, pr)))
        LOG.info("%-20s CV AUC %.4f test AUC %.4f F1 %.4f (%.1fs)", name, cv.mean(), rows[-1]["ROC_AUC"], rows[-1]["F1"], time.time() - t)

    grids = {"Random Forest": (ml.RandomForestClassifier(class_weight="balanced", random_state=SEED), {"n_estimators": [40, 60], "max_depth": [4, 6, 8], "min_samples_leaf": [30, 60, 120], "max_features": ["sqrt", 0.5]}),
             "Gradient Boosting": (ml.GradientBoostingClassifier(class_weight="balanced", random_state=SEED), {"n_estimators": [50, 100, 150], "learning_rate": [0.02, 0.05, 0.1], "max_depth": [2, 3], "min_samples_leaf": [30, 80]}),
             "Decision Tree": (ml.DecisionTreeClassifier(class_weight="balanced", random_state=SEED), {"max_depth": [3, 4, 5, 7], "min_samples_leaf": [30, 60, 120]})}
    d0 = pd.DataFrame(rows[1:]); top2 = d0[d0.Model.isin(list(grids))].sort_values("CV_AUC", ascending=False).Model.head(2).tolist()
    tuned = {}
    for name in top2:
        est, g = grids[name]; s = ml.RandomizedSearchCV(est, g, n_iter=4, cv=3, scoring="roc_auc", random_state=SEED).fit(Xtr.values, ytr)
        pr = s.best_estimator_.predict_proba(Xte.values)[:, 1]; tuned[name + " (tuned)"] = s.best_estimator_; probas[name + " (tuned)"] = pr
        rows.append(dict(Model=name + " (tuned)", CV_AUC=s.best_score_, Stage="tuned", Params=str(s.best_params_), **metrics(yte, pr)))
        LOG.info("%s tuned %s CV AUC %.4f test AUC %.4f", name, s.best_params_, s.best_score_, rows[-1]["ROC_AUC"])
    res = pd.DataFrame(rows)
    sel = res[res.Stage != "baseline"].sort_values("CV_AUC", ascending=False).iloc[0]
    best_name = sel.Model; best = tuned.get(best_name, fitted.get(best_name)); pr = probas[best_name]
    is_lin = best_name == "Logistic Regression"
    LOG.info("selected by CV AUC: %s", best_name)

    perm = permutation_auc(best, Xte, yte, scaler=sc if is_lin else None)
    imp = pd.DataFrame({"feature": X.columns, "auc_drop_when_shuffled": perm.values}).sort_values("auc_drop_when_shuffled", ascending=False)

    allp = best.predict_proba(X.values)[:, 1]   # (Scaled wrapper handles standardisation for the linear model)
    sc_df = ch[["CustomerID", "Recency", "Frequency", "Monetary", "Churn"]].copy(); sc_df["ChurnProbability"] = allp
    sc_df["Split"] = "train"; sc_df.loc[sc_df.index[ite], "Split"] = "test"
    q = sc_df["ChurnProbability"].quantile([.33, .66]).values
    sc_df["RiskSegment"] = np.where(sc_df.ChurnProbability >= q[1], "High", np.where(sc_df.ChurnProbability >= q[0], "Medium", "Low"))

    out = ROOT / "data" / "cleaned"
    res.to_csv(out / "model_comparison_churn.csv", index=False); imp.to_csv(out / "feature_importance_churn.csv", index=False)
    sc_df.to_csv(out / "churn_scores.csv", index=False); ch.to_csv(out / "churn_features.csv", index=False)
    with open(ROOT / "models" / "churn_best_model.pkl", "wb") as f:
        pickle.dump(dict(model=best, name=best_name, features=list(X.columns), scaler=sc if is_lin else None, cutoff=str(cutoff.date()),
                         backend=ml.BACKEND, test_metrics=metrics(yte, pr)), f)

    cm = ml.confusion_matrix(yte, (pr >= .5).astype(int)); fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    ax[0].imshow(cm, cmap="Reds"); [ax[0].text(j, i, f"{cm[i, j]}", ha="center", va="center", fontsize=14) for i in range(2) for j in range(2)]
    ax[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Retained", "Churned"], yticklabels=["Retained", "Churned"], xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix - {best_name}")
    for n, p in probas.items():
        fpr, tpr, _ = ml.roc_curve(yte, p); ax[1].plot(fpr, tpr, label=f"{n} (AUC {ml.roc_auc_score(yte, p):.3f})", lw=1.3)
    ax[1].plot([0, 1], [0, 1], "k--", label="random"); ax[1].set(xlabel="False positive rate", ylabel="True positive rate", title="ROC curves (test set)"); ax[1].legend(fontsize=7)
    t10 = imp.head(10).iloc[::-1]; ax[2].barh(t10.feature, t10.auc_drop_when_shuffled, color="#E23744"); ax[2].set(xlabel="Drop in ROC-AUC when shuffled", title="Top-10 churn drivers (permutation)")
    plt.tight_layout(); plt.savefig(img / "ml_churn_evaluation.png", dpi=130); plt.close()
    print(res.round(4).to_string(index=False)); print("\nTop drivers:\n", imp.head(10).round(4).to_string(index=False))


if __name__ == "__main__":
    main()
