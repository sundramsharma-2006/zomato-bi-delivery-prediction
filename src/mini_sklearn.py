"""mini_sklearn.py - a small NumPy re-implementation of the scikit-learn API subset used here.

WHY THIS EXISTS: the environment in which this project was first executed had no
network access and no scikit-learn.  model_delivery.py / model_churn.py do

    try:  import sklearn ...      # preferred: real scikit-learn
    except ImportError: from . import mini_sklearn as ...

so on YOUR machine (pip install -r requirements.txt) the real library is used and
every number/plot/model can be regenerated with `python run_pipeline.py`.
Implemented: LinearRegression, DecisionTree/RandomForest/GradientBoosting
(Regressor + binary Classifier), LogisticRegression, StandardScaler,
train_test_split, KFold/StratifiedKFold, cross_val_score, GridSearchCV,
RandomizedSearchCV, and the metrics used in the PRD.

Trees are histogram-based (<=32 quantile bins per feature), weighted, and use
variance reduction (== gini for 0/1 targets).
"""
import inspect
import itertools
import numpy as np

# ----------------------------------------------------------------------------- base
class BaseEstimator:
    def get_params(self):
        sig = inspect.signature(self.__init__)
        return {k: getattr(self, k) for k in sig.parameters if k != "self"}

    def set_params(self, **p):
        for k, v in p.items():
            setattr(self, k, v)
        return self


def clone(est):
    return est.__class__(**est.get_params())


def _arr(X):
    return np.asarray(X, dtype=float)


# ----------------------------------------------------------------------------- preprocessing
class StandardScaler:
    def fit(self, X):
        X = _arr(X); self.mean_ = X.mean(0); self.scale_ = X.std(0); self.scale_[self.scale_ == 0] = 1.0
        return self

    def transform(self, X):
        return (_arr(X) - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


def train_test_split(*arrays, test_size=0.2, random_state=None, stratify=None):
    n = len(arrays[0]); rng = np.random.RandomState(random_state)
    if stratify is None:
        idx = rng.permutation(n); nt = int(round(n * test_size))
        te, tr = idx[:nt], idx[nt:]
    else:
        st = np.asarray(stratify); te = []
        for k in np.unique(st):
            ii = rng.permutation(np.where(st == k)[0]); te.append(ii[:int(round(len(ii) * test_size))])
        te = np.concatenate(te); tr = np.setdiff1d(np.arange(n), te); rng.shuffle(tr); rng.shuffle(te)
    out = []
    for a in arrays:
        a_ = a.iloc if hasattr(a, "iloc") else a
        a_ = np.asarray(a) if not hasattr(a, "iloc") else a
        out += [a.iloc[tr] if hasattr(a, "iloc") else a_[tr], a.iloc[te] if hasattr(a, "iloc") else a_[te]]
    return out


class KFold:
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits, self.shuffle, self.random_state = n_splits, shuffle, random_state

    def split(self, X, y=None):
        n = len(X); idx = np.arange(n)
        if self.shuffle:
            np.random.RandomState(self.random_state).shuffle(idx)
        for f in np.array_split(idx, self.n_splits):
            yield np.setdiff1d(idx, f, assume_unique=True), f


class StratifiedKFold(KFold):
    def split(self, X, y):
        y = np.asarray(y); rng = np.random.RandomState(self.random_state)
        folds = [[] for _ in range(self.n_splits)]
        for k in np.unique(y):
            ii = np.where(y == k)[0]; rng.shuffle(ii)
            for j, part in enumerate(np.array_split(ii, self.n_splits)):
                folds[j] += part.tolist()
        allidx = np.arange(len(y))
        for f in folds:
            f = np.array(sorted(f)); yield np.setdiff1d(allidx, f), f


# ----------------------------------------------------------------------------- metrics
def mean_absolute_error(y, p): return float(np.mean(np.abs(np.asarray(y) - np.asarray(p))))
def mean_squared_error(y, p): return float(np.mean((np.asarray(y) - np.asarray(p)) ** 2))
def r2_score(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    ss = ((y - y.mean()) ** 2).sum()
    return float(1 - ((y - p) ** 2).sum() / ss) if ss > 0 else 0.0
def accuracy_score(y, p): return float(np.mean(np.asarray(y) == np.asarray(p)))
def confusion_matrix(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return np.array([[np.sum((y == 0) & (p == 0)), np.sum((y == 0) & (p == 1))],
                     [np.sum((y == 1) & (p == 0)), np.sum((y == 1) & (p == 1))]])
def precision_score(y, p, zero_division=0):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel(); return float(tp / (tp + fp)) if tp + fp else zero_division
def recall_score(y, p, zero_division=0):
    tn, fp, fn, tp = confusion_matrix(y, p).ravel(); return float(tp / (tp + fn)) if tp + fn else zero_division
def f1_score(y, p, zero_division=0):
    pr, rc = precision_score(y, p), recall_score(y, p); return float(2 * pr * rc / (pr + rc)) if pr + rc else zero_division
def roc_curve(y, s):
    y, s = np.asarray(y), np.asarray(s); o = np.argsort(-s, kind="mergesort"); y, s = y[o], s[o]
    tps = np.cumsum(y); fps = np.cumsum(1 - y); d = np.where(np.diff(s))[0]; t = np.r_[d, len(y) - 1]
    tpr = np.r_[0, tps[t] / max(tps[-1], 1)]; fpr = np.r_[0, fps[t] / max(fps[-1], 1)]
    return fpr, tpr, np.r_[np.inf, s[t]]
def roc_auc_score(y, s):
    fpr, tpr, _ = roc_curve(y, s); return float(np.trapz(tpr, fpr))
def average_precision_score(y, s):
    y, s = np.asarray(y), np.asarray(s); o = np.argsort(-s, kind="mergesort"); y = y[o]
    tp = np.cumsum(y); prec = tp / np.arange(1, len(y) + 1); return float((prec * y).sum() / max(y.sum(), 1))

_SCORERS = {"r2": (r2_score, 1), "neg_mean_absolute_error": (mean_absolute_error, -1),
            "neg_root_mean_squared_error": (lambda y, p: np.sqrt(mean_squared_error(y, p)), -1),
            "f1": (f1_score, 1), "accuracy": (accuracy_score, 1)}

def _score(est, X, y, scoring):
    if scoring == "roc_auc":
        return roc_auc_score(y, est.predict_proba(X)[:, 1])
    fn, sign = _SCORERS[scoring]
    return sign * fn(y, est.predict(X))


# ----------------------------------------------------------------------------- linear models
class LinearRegression(BaseEstimator):
    def __init__(self): pass
    def fit(self, X, y, sample_weight=None):
        X = _arr(X); A = np.c_[np.ones(len(X)), X]
        self.coef_all = np.linalg.lstsq(A, np.asarray(y, float), rcond=None)[0]
        self.intercept_, self.coef_ = self.coef_all[0], self.coef_all[1:]; return self
    def predict(self, X): return _arr(X) @ self.coef_ + self.intercept_


class LogisticRegression(BaseEstimator):
    """L2-regularised logistic regression solved with Newton/IRLS (inputs should be scaled)."""
    def __init__(self, C=1.0, class_weight=None, max_iter=100, random_state=None):
        self.C, self.class_weight, self.max_iter, self.random_state = C, class_weight, max_iter, random_state
    def fit(self, X, y, sample_weight=None):
        X = _arr(X); y = np.asarray(y, float); A = np.c_[np.ones(len(X)), X]
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if self.class_weight == "balanced":
            n1 = y.sum(); n0 = len(y) - n1; w = w * np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))
        lam = 1.0 / self.C; reg = np.eye(A.shape[1]) * lam; reg[0, 0] = 0
        b = np.zeros(A.shape[1])
        for _ in range(self.max_iter):
            p = 1 / (1 + np.exp(-A @ b)); g = A.T @ (w * (p - y)) + reg @ b
            H = (A * (w * p * (1 - p))[:, None]).T @ A + reg + 1e-8 * np.eye(A.shape[1])
            step = np.linalg.solve(H, g); b -= step
            if np.abs(step).max() < 1e-7: break
        self.intercept_, self.coef_ = b[0], b[1:]; return self
    def decision_function(self, X): return _arr(X) @ self.coef_ + self.intercept_
    def predict_proba(self, X):
        p = 1 / (1 + np.exp(-self.decision_function(X))); return np.c_[1 - p, p]
    def predict(self, X): return (self.decision_function(X) > 0).astype(int)


# ----------------------------------------------------------------------------- trees
NBINS = 32

def _make_bins(X):
    edges = []
    for j in range(X.shape[1]):
        u = np.unique(X[:, j])
        if len(u) <= NBINS: e = (u[:-1] + u[1:]) / 2
        else: e = np.unique(np.quantile(X[:, j], np.linspace(0, 1, NBINS + 1)[1:-1]))
        edges.append(e)
    return edges

def _bin(X, edges):
    return np.column_stack([np.searchsorted(edges[j], X[:, j], side="right") for j in range(X.shape[1])]).astype(np.int16)


class _Tree:
    """Weighted regression tree on pre-binned data. value = weighted mean of target."""
    def __init__(self, max_depth=6, min_samples_leaf=5, max_features=None, rng=None):
        self.max_depth = max_depth if max_depth else 30
        self.msl, self.max_features, self.rng = min_samples_leaf, max_features, rng or np.random.RandomState(0)

    def fit(self, Xb, y, w, nfeat_total, nb):
        n, F = Xb.shape; self.F = F
        self.feat, self.thr, self.left, self.right, self.val = [], [], [], [], []
        self.importance = np.zeros(F)
        def new(v):
            self.feat.append(-1); self.thr.append(-1); self.left.append(-1); self.right.append(-1); self.val.append(v); return len(self.val) - 1
        sw = w.sum(); root = new((w * y).sum() / sw if sw > 0 else 0.0)
        stack = [(root, np.arange(n), 0)]
        while stack:
            node, idx, depth = stack.pop()
            if depth >= self.max_depth or len(idx) < 2 * self.msl: continue
            wi, yi = w[idx], y[idx]; W = wi.sum(); S = (wi * yi).sum()
            if W <= 0: continue
            if self.max_features and self.max_features < F:        # sample candidate features per node (random-forest style)
                cols = np.sort(self.rng.choice(F, self.max_features, replace=False))
            else:
                cols = np.arange(F)
            Fc = len(cols); offs_c = (np.arange(Fc) * nb)[None, :]
            flat = (Xb[idx][:, cols] + offs_c).ravel()
            cw = np.bincount(flat, weights=np.repeat(wi, Fc), minlength=Fc * nb).reshape(Fc, nb)
            cs = np.bincount(flat, weights=np.repeat(wi * yi, Fc), minlength=Fc * nb).reshape(Fc, nb)
            cn = np.bincount(flat, minlength=Fc * nb).reshape(Fc, nb)
            Wl, Sl, Nl = cw.cumsum(1)[:, :-1], cs.cumsum(1)[:, :-1], cn.cumsum(1)[:, :-1]
            Wr, Sr, Nr = W - Wl, S - Sl, len(idx) - Nl
            ok = (Nl >= self.msl) & (Nr >= self.msl) & (Wl > 0) & (Wr > 0)
            if not ok.any(): continue
            gain = np.where(ok, Sl ** 2 / np.where(Wl > 0, Wl, 1) + Sr ** 2 / np.where(Wr > 0, Wr, 1) - S ** 2 / W, -np.inf)
            fi, b = np.unravel_index(np.argmax(gain), gain.shape)
            if gain[fi, b] <= 1e-12: continue
            f = int(cols[fi])
            go = Xb[idx, f] <= b; li, ri = idx[go], idx[~go]
            self.feat[node], self.thr[node] = int(f), int(b); self.importance[f] += gain[fi, b]
            l = new((w[li] * y[li]).sum() / max(w[li].sum(), 1e-12)); r = new((w[ri] * y[ri]).sum() / max(w[ri].sum(), 1e-12))
            self.left[node], self.right[node] = l, r
            stack += [(l, li, depth + 1), (r, ri, depth + 1)]
        self.feat, self.thr = np.array(self.feat), np.array(self.thr)
        self.left, self.right, self.val = np.array(self.left), np.array(self.right), np.array(self.val, float)
        return self

    def apply(self, Xb):
        pos = np.zeros(len(Xb), int)
        for _ in range(self.max_depth + 1):
            f = self.feat[pos]; internal = f >= 0
            if not internal.any(): break
            go_left = Xb[np.arange(len(Xb)), np.where(internal, f, 0)] <= self.thr[pos]
            pos = np.where(internal, np.where(go_left, self.left[pos], self.right[pos]), pos)
        return pos

    def predict(self, Xb): return self.val[self.apply(Xb)]


class _TreeEnsembleBase(BaseEstimator):
    def _prep(self, X):
        X = _arr(X); self.edges_ = _make_bins(X); return _bin(X, self.edges_)
    def _mf(self, F):
        mf = self.max_features
        if mf is None: return None
        if mf == "sqrt": return max(1, int(np.sqrt(F)))
        if isinstance(mf, float): return max(1, int(mf * F))
        return int(mf)
    def _cw(self, y, sample_weight):
        w = np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float)
        if getattr(self, "class_weight", None) == "balanced":
            n1 = y.sum(); n0 = len(y) - n1; w = w * np.where(y == 1, len(y) / (2 * n1), len(y) / (2 * n0))
        return w


class _SingleTree(_TreeEnsembleBase):
    def fit(self, X, y, sample_weight=None):
        Xb = self._prep(X); y = np.asarray(y, float); w = self._cw(y, sample_weight)
        self.tree_ = _Tree(self.max_depth, self.min_samples_leaf, None, np.random.RandomState(self.random_state)).fit(Xb, y, w, Xb.shape[1], NBINS + 1)
        s = self.tree_.importance.sum(); self.feature_importances_ = self.tree_.importance / s if s else self.tree_.importance
        return self
    def _raw(self, X): return self.tree_.predict(_bin(_arr(X), self.edges_))

class DecisionTreeRegressor(_SingleTree):
    def __init__(self, max_depth=None, min_samples_leaf=1, random_state=None):
        self.max_depth, self.min_samples_leaf, self.random_state = max_depth, min_samples_leaf, random_state
    def predict(self, X): return self._raw(X)

class DecisionTreeClassifier(_SingleTree):
    def __init__(self, max_depth=None, min_samples_leaf=1, class_weight=None, random_state=None):
        self.max_depth, self.min_samples_leaf, self.class_weight, self.random_state = max_depth, min_samples_leaf, class_weight, random_state
    def predict_proba(self, X): p = np.clip(self._raw(X), 0, 1); return np.c_[1 - p, p]
    def predict(self, X): return (self._raw(X) >= 0.5).astype(int)


class _Forest(_TreeEnsembleBase):
    def fit(self, X, y, sample_weight=None):
        Xb = self._prep(X); y = np.asarray(y, float); n, F = Xb.shape; rng = np.random.RandomState(self.random_state)
        base_w = self._cw(y, sample_weight); self.trees_ = []; imp = np.zeros(F)
        for _ in range(self.n_estimators):
            bs = rng.randint(0, n, n); w = np.bincount(bs, minlength=n) * base_w
            t = _Tree(self.max_depth, self.min_samples_leaf, self._mf(F), np.random.RandomState(rng.randint(1 << 30))).fit(Xb, y, w, F, NBINS + 1)
            self.trees_.append(t); s = t.importance.sum(); imp += t.importance / s if s else 0
        self.feature_importances_ = imp / imp.sum() if imp.sum() else imp; return self
    def _raw(self, X):
        Xb = _bin(_arr(X), self.edges_); return np.mean([t.predict(Xb) for t in self.trees_], axis=0)

class RandomForestRegressor(_Forest):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features=1.0, random_state=None, n_jobs=None):
        self.n_estimators, self.max_depth, self.min_samples_leaf, self.max_features, self.random_state, self.n_jobs = n_estimators, max_depth, min_samples_leaf, max_features, random_state, n_jobs
    def predict(self, X): return self._raw(X)

class RandomForestClassifier(_Forest):
    def __init__(self, n_estimators=100, max_depth=None, min_samples_leaf=1, max_features="sqrt", class_weight=None, random_state=None, n_jobs=None):
        self.n_estimators, self.max_depth, self.min_samples_leaf, self.max_features, self.class_weight, self.random_state, self.n_jobs = n_estimators, max_depth, min_samples_leaf, max_features, class_weight, random_state, n_jobs
    def predict_proba(self, X): p = np.clip(self._raw(X), 0, 1); return np.c_[1 - p, p]
    def predict(self, X): return (self._raw(X) >= 0.5).astype(int)


class _GB(_TreeEnsembleBase):
    def _boost(self, X, y, w, classify):
        Xb = self._prep(X); y = np.asarray(y, float); n, F = Xb.shape; rng = np.random.RandomState(self.random_state)
        if classify:
            p0 = np.clip((w * y).sum() / w.sum(), 1e-6, 1 - 1e-6); self.init_ = np.log(p0 / (1 - p0))
        else: self.init_ = (w * y).sum() / w.sum()
        Fm = np.full(n, self.init_); self.trees_ = []; imp = np.zeros(F)
        for _ in range(self.n_estimators):
            if classify: p = 1 / (1 + np.exp(-Fm)); res = y - p
            else: res = y - Fm
            sub = np.ones(n) if self.subsample >= 1 else (rng.rand(n) < self.subsample).astype(float)
            t = _Tree(self.max_depth, self.min_samples_leaf, None, rng).fit(Xb, res, w * sub, F, NBINS + 1)
            leaf = t.apply(Xb)
            if classify:   # Newton leaf values
                num = np.bincount(leaf, weights=w * res, minlength=len(t.val)); den = np.bincount(leaf, weights=w * p * (1 - p), minlength=len(t.val))
                t.val = np.where(den > 1e-12, num / np.maximum(den, 1e-12), 0.0)
            Fm = Fm + self.learning_rate * t.val[leaf]; self.trees_.append(t); s = t.importance.sum(); imp += t.importance / s if s else 0
        self.feature_importances_ = imp / imp.sum() if imp.sum() else imp
    def _F(self, X):
        Xb = _bin(_arr(X), self.edges_); out = np.full(len(Xb), self.init_)
        for t in self.trees_: out += self.learning_rate * t.predict(Xb)
        return out

class GradientBoostingRegressor(_GB):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=1, subsample=1.0, random_state=None):
        self.n_estimators, self.learning_rate, self.max_depth, self.min_samples_leaf, self.subsample, self.random_state = n_estimators, learning_rate, max_depth, min_samples_leaf, subsample, random_state
    def fit(self, X, y, sample_weight=None):
        self._boost(X, y, np.ones(len(y)) if sample_weight is None else np.asarray(sample_weight, float), False); return self
    def predict(self, X): return self._F(X)

class GradientBoostingClassifier(_GB):
    def __init__(self, n_estimators=100, learning_rate=0.1, max_depth=3, min_samples_leaf=1, subsample=1.0, class_weight=None, random_state=None):
        self.n_estimators, self.learning_rate, self.max_depth, self.min_samples_leaf, self.subsample, self.class_weight, self.random_state = n_estimators, learning_rate, max_depth, min_samples_leaf, subsample, class_weight, random_state
    def fit(self, X, y, sample_weight=None):
        y = np.asarray(y, float); self._boost(X, y, self._cw(y, sample_weight), True); return self
    def predict_proba(self, X): p = 1 / (1 + np.exp(-self._F(X))); return np.c_[1 - p, p]
    def predict(self, X): return (self._F(X) > 0).astype(int)


# ----------------------------------------------------------------------------- model selection
def cross_val_score(est, X, y, cv=5, scoring="r2", random_state=0):
    X = _arr(X); y = np.asarray(y)
    splitter = StratifiedKFold(cv, True, random_state) if scoring in ("f1", "roc_auc", "accuracy") else KFold(cv, True, random_state)
    out = []
    for tr, te in splitter.split(X, y):
        m = clone(est).fit(X[tr], y[tr]); out.append(_score(m, X[te], y[te], scoring))
    return np.array(out)


class _Search:
    def _candidates(self): raise NotImplementedError
    def fit(self, X, y):
        X = _arr(X); y = np.asarray(y); self.cv_results_ = {"params": [], "mean_test_score": []}
        for p in self._candidates():
            s = cross_val_score(clone(self.estimator).set_params(**p), X, y, self.cv, self.scoring).mean()
            self.cv_results_["params"].append(p); self.cv_results_["mean_test_score"].append(float(s))
        i = int(np.argmax(self.cv_results_["mean_test_score"]))
        self.best_params_, self.best_score_ = self.cv_results_["params"][i], self.cv_results_["mean_test_score"][i]
        self.best_estimator_ = clone(self.estimator).set_params(**self.best_params_).fit(X, y); return self
    def predict(self, X): return self.best_estimator_.predict(X)

class GridSearchCV(_Search):
    def __init__(self, estimator, param_grid, cv=3, scoring="r2", n_jobs=None):
        self.estimator, self.param_grid, self.cv, self.scoring = estimator, param_grid, cv, scoring
    def _candidates(self):
        ks = list(self.param_grid); return [dict(zip(ks, v)) for v in itertools.product(*[self.param_grid[k] for k in ks])]

class RandomizedSearchCV(_Search):
    def __init__(self, estimator, param_distributions, n_iter=8, cv=3, scoring="r2", random_state=0, n_jobs=None):
        self.estimator, self.param_distributions, self.n_iter, self.cv, self.scoring, self.random_state = estimator, param_distributions, n_iter, cv, scoring, random_state
    def _candidates(self):
        rng = np.random.RandomState(self.random_state); ks = list(self.param_distributions)
        allc = [dict(zip(ks, v)) for v in itertools.product(*[self.param_distributions[k] for k in ks])]
        sel = rng.permutation(len(allc))[:self.n_iter]; return [allc[i] for i in sel]
