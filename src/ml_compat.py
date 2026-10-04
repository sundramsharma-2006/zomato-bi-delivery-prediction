"""ml_compat.py - single import point for ML tooling.

Uses real scikit-learn when installed; otherwise falls back to the NumPy
re-implementation in mini_sklearn.py (same API subset). `BACKEND` records which
one produced the results so it can be written into the reports.
"""
try:
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
    from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                                  GradientBoostingRegressor, GradientBoostingClassifier)
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import (train_test_split, cross_val_score, RandomizedSearchCV,
                                         GridSearchCV, KFold, StratifiedKFold)
    from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score, accuracy_score,
                                 precision_score, recall_score, f1_score, roc_auc_score, roc_curve,
                                 confusion_matrix, average_precision_score)
    from sklearn.base import clone
    import sklearn
    BACKEND = f"scikit-learn {sklearn.__version__}"
except ImportError:                                   # pragma: no cover - offline fallback
    from .mini_sklearn import *                       # noqa: F401,F403
    from .mini_sklearn import clone
    BACKEND = "mini_sklearn (NumPy fallback, scikit-learn API subset)"
