from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def dt_metrics(positive_scores: np.ndarray, negative_scores: np.ndarray) -> tuple[float, float]:
    scores = np.concatenate((positive_scores.reshape(-1), negative_scores.reshape(-1)))
    labels = np.concatenate((np.ones(positive_scores.size, dtype=np.uint8),
                             np.zeros(negative_scores.size, dtype=np.uint8)))
    return float(roc_auc_score(labels, scores)), float(average_precision_score(labels, scores))


def grouped_df_metrics(requested_scores: np.ndarray,
                       retained_groups: np.ndarray) -> tuple[float, float]:
    """Mean requested-vs-retained ranking metrics over the supplied control groups."""
    requested = np.asarray(requested_scores).reshape(-1)
    retained = np.asarray(retained_groups)
    if retained.ndim != 2 or retained.shape[0] != 500:
        raise ValueError("formal Df evaluation requires exactly 500 retained-control groups")
    labels = np.concatenate((np.zeros(requested.size, dtype=np.uint8),
                             np.ones(retained.shape[1], dtype=np.uint8)))
    aucs, aps = [], []
    for group in retained:
        scores = np.concatenate((requested, group))
        aucs.append(roc_auc_score(labels, scores))
        aps.append(average_precision_score(labels, scores))
    return float(np.mean(aucs)), float(np.mean(aps))


def grouped_df_metrics_vectorized(requested_scores: np.ndarray,
                                  retained_groups: np.ndarray) -> tuple[float, float]:
    """Vectorized over the same 500 groups with tie handling matching sklearn."""
    requested = np.asarray(requested_scores).reshape(-1)
    retained = np.asarray(retained_groups)
    if retained.ndim != 2 or retained.shape[0] != 500:
        raise ValueError("formal Df evaluation requires exactly 500 retained-control groups")
    n_groups, n_positive = retained.shape
    ordered = np.sort(requested, kind="mergesort")
    left = np.searchsorted(ordered, retained, side="left")
    right = np.searchsorted(ordered, retained, side="right")
    auc = float(((left + 0.5 * (right - left)).mean(axis=1) / requested.size).mean())

    scores = np.concatenate((np.broadcast_to(requested, (n_groups, requested.size)), retained), axis=1)
    labels = np.concatenate((np.zeros((n_groups, requested.size), dtype=np.uint8),
                             np.ones((n_groups, n_positive), dtype=np.uint8)), axis=1)
    order = np.argsort(-scores, axis=1, kind="mergesort")
    sorted_scores = np.take_along_axis(scores, order, axis=1)
    sorted_labels = np.take_along_axis(labels, order, axis=1)
    tp = np.cumsum(sorted_labels, axis=1, dtype=np.float64)
    before = np.concatenate((np.zeros((n_groups, 1), dtype=np.float64), tp[:, :-1]), axis=1)
    ranks = np.arange(1, scores.shape[1] + 1, dtype=np.float64)[None, :]
    last = np.ones_like(sorted_scores, dtype=bool)
    last[:, :-1] = sorted_scores[:, :-1] != sorted_scores[:, 1:]
    first = np.ones_like(sorted_scores, dtype=bool)
    first[:, 1:] = sorted_scores[:, 1:] != sorted_scores[:, :-1]
    prior_tp = np.maximum.accumulate(np.where(first, before, 0.0), axis=1)
    ap_per_group = (((tp - prior_tp) * tp / ranks) * last).sum(axis=1) / n_positive
    return auc, float(ap_per_group.mean())
