"""Held-out scores and exploratory t-SNE; embedding is never a training target."""

import numpy as np
from scipy.stats import spearmanr
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import mean_absolute_error, r2_score

from .features import pair_channels


def scores(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.ndim != 1 or y.shape != prediction.shape or not len(y) or not np.isfinite(y).all() or not np.isfinite(prediction).all():
        raise ValueError("Aligned finite targets and predictions required")
    variable = len(y) >= 2 and np.ptp(y) > 0
    return {"n": len(y), "MAE": float(mean_absolute_error(y, prediction)),
            "R2": float(r2_score(y, prediction)) if variable else None,
            "spearman": float(spearmanr(y, prediction).statistic) if variable and np.ptp(prediction) > 0 else None}


def distance_matrix(mixtures, model):
    n = len(mixtures)
    if n < 4 or n > 2000:
        raise ValueError("Embedding supports 4..2000 mixtures; subset larger data explicitly")
    d = np.zeros((n, n))
    for i in range(n - 1):
        channels = np.stack([pair_channels(mixtures[i], mixtures[j]) for j in range(i + 1, n)])
        d[i, i + 1:] = model.predict(channels)
        d[i + 1:, i] = d[i, i + 1:]
    return d


def tsne(distance, seed=42, perplexity=30):
    d = np.asarray(distance, float)
    if d.ndim != 2 or d.shape[0] != d.shape[1] or len(d) < 4 or not np.isfinite(d).all() or (d < 0).any() or not np.allclose(d, d.T) or not np.allclose(np.diag(d), 0):
        raise ValueError("t-SNE needs a finite symmetric nonnegative square distance matrix with zero diagonal")
    if not np.isfinite(perplexity) or perplexity <= 0:
        raise ValueError("Perplexity must be positive")
    if not d.max():
        raise ValueError("All mixtures have zero distance; t-SNE would suggest artificial structure")
    p = min(float(perplexity), (len(d) - 1) / 3)
    xy = TSNE(metric="precomputed", init="random", random_state=seed, perplexity=p, learning_rate="auto", max_iter=1000, n_jobs=1).fit_transform(d)
    k = min(10, max(1, (len(d) - 1) // 3))
    return xy, {"seed": seed, "perplexity": p, "iterations": 1000, "trustworthiness_k": k,
                "trustworthiness": float(trustworthiness(d, xy, metric="precomputed", n_neighbors=k)),
                "interpretation": "Exploratory local neighborhoods, not comparable axes or quantitative perceptual distances"}
