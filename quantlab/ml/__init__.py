"""
quantlab.ml
===========

Machine-learning toolkit for trading: leak-free features, triple-barrier
labeling, purged cross-validation, gradient-boosted models, and Optuna
optimization of both strategy settings and model hyperparameters.
"""

from .cv import PurgedKFold
from .features import DEFAULT_FEATURES, assemble_dataset, make_features
from .labeling import binary_labels, triple_barrier_labels
from .meta import apply_meta, meta_dataset, triple_barrier_meta
from .pipeline import (
    feature_importances,
    fit_full_model,
    make_model,
    oos_signal,
    walk_forward_predict,
)

__all__ = [
    "make_features",
    "assemble_dataset",
    "DEFAULT_FEATURES",
    "triple_barrier_labels",
    "binary_labels",
    "PurgedKFold",
    # walk-forward ML pipeline
    "make_model",
    "walk_forward_predict",
    "fit_full_model",
    "feature_importances",
    "oos_signal",
    # meta-labeling
    "triple_barrier_meta",
    "meta_dataset",
    "apply_meta",
]
