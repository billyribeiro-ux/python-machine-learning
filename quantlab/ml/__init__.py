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

__all__ = [
    "make_features",
    "assemble_dataset",
    "DEFAULT_FEATURES",
    "triple_barrier_labels",
    "binary_labels",
    "PurgedKFold",
]
