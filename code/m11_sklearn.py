"""Module 11 — leak-free ML: features, triple-barrier labels, purged CV.

Run:  python code/m11_sklearn.py
"""
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from quantlab.data import get_provider
from quantlab.ml import assemble_dataset, triple_barrier_labels, PurgedKFold

df = get_provider("yahoo").get_ohlcv("SPY", start="2010-01-01")
close = df["close"]

# Triple-barrier labels: +1 if +2 sigma hit before -2 sigma within 10 bars.
lab = triple_barrier_labels(close, horizon=10, upper=2.0, lower=2.0)
y = (lab["label"] > 0).astype(float).where(lab["label"].notna())

X, y = assemble_dataset(df, y)
print("Dataset:", X.shape, "| positive label rate:", f"{y.mean():.1%}")

# A pipeline so scaling is fit ONLY on training folds (avoids a subtle leak).
model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))

# Purged + embargoed K-fold — honest CV for overlapping, autocorrelated labels.
pk = PurgedKFold(n_splits=5, embargo=0.01, label_horizon=10)
aucs = []
for tr, te in pk.split(X):
    model.fit(X.iloc[tr], y.iloc[tr])
    p = model.predict_proba(X.iloc[te])[:, 1]
    if len(np.unique(y.iloc[te])) > 1:
        aucs.append(roc_auc_score(y.iloc[te], p))

print(f"Purged CV AUC: mean={np.mean(aucs):.3f}  folds={[round(a,3) for a in aucs]}")
print("AUC near 0.5 is the honest, common result — markets are hard. The point "
      "is a pipeline that won't fool you, not a magic number.")
