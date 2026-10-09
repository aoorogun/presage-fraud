"""Model Family A (supervised tabular) and B (anomaly detection) wrappers with a common interface."""
from __future__ import annotations
import time
import numpy as np
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.pipeline import make_pipeline


def make_supervised(name: str, pos_weight: float, seed: int = 42, n_jobs: int = 2):
    """Return an unfitted classifier with class-imbalance handled by weighting (no resampling before splitting)."""
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=300, class_weight={0: 1.0, 1: min(pos_weight, 200)}, C=0.5))
    if name == "random_forest":
        return RandomForestClassifier(n_estimators=250, min_samples_leaf=3, max_features="sqrt", class_weight="balanced_subsample", n_jobs=n_jobs, random_state=seed)
    if name == "xgboost":
        import xgboost as xgb
        return xgb.XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=min(pos_weight, 50),
                                 eval_metric="aucpr", tree_method="hist", n_jobs=n_jobs, random_state=seed, verbosity=0)
    if name == "lightgbm":
        import lightgbm as lgb
        # NOTE: scale_pos_weight combined with bagging diverged in our runs (train PR-AUC ~0.004), so LightGBM uses
        # unweighted loss; imbalance is handled downstream by threshold selection on validation data.
        return lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31, colsample_bytree=0.8, min_child_samples=20,
                                  n_jobs=1, random_state=seed, verbose=-1)   # n_jobs=1: avoids an OpenMP clash with xgboost on macOS
    if name == "catboost":
        from catboost import CatBoostClassifier
        return CatBoostClassifier(iterations=500, depth=6, learning_rate=0.06, scale_pos_weight=min(pos_weight, 50), thread_count=n_jobs, random_seed=seed, verbose=False)
    raise ValueError(name)


class AnomalyModel:
    """Unsupervised detectors trained on (mostly legitimate) training rows. score = higher means more anomalous."""

    def __init__(self, name: str, seed: int = 42, max_fit: int = 40000):
        self.name, self.seed, self.max_fit = name, seed, max_fit

    def fit(self, X: np.ndarray):
        rng = np.random.default_rng(self.seed)
        sub = X[rng.choice(len(X), min(len(X), self.max_fit), replace=False)]
        self.scaler = StandardScaler().fit(sub)
        Z = np.clip(self.scaler.transform(sub), -8, 8)
        if self.name == "isolation_forest":
            self.m = IsolationForest(n_estimators=300, contamination="auto", random_state=self.seed, n_jobs=2).fit(Z)
        elif self.name == "lof":
            self.m = LocalOutlierFactor(n_neighbors=35, novelty=True, n_jobs=2).fit(Z[:20000])
        elif self.name == "ocsvm":
            self.m = OneClassSVM(nu=0.01, gamma="scale").fit(Z[:8000])
        elif self.name in ("autoencoder", "vae"):
            from presage.models.autoencoder import fit_autoencoder
            self.m = fit_autoencoder(Z, variational=(self.name == "vae"), seed=self.seed)
        return self

    def score(self, X: np.ndarray) -> np.ndarray:
        Z = np.clip(self.scaler.transform(X), -8, 8)
        if self.name == "isolation_forest": return -self.m.score_samples(Z)
        if self.name in ("lof", "ocsvm"):
            out = np.empty(len(Z))
            for i in range(0, len(Z), 20000): out[i:i + 20000] = -self.m.score_samples(Z[i:i + 20000])
            return out
        return self.m.score(Z)
