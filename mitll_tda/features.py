"""Topological feature extraction and feature-space projection.

Mirrors the notebooks' ``extract_topological_features`` (per-sample summary
statistics of a persistence diagram) and ``plot_topological_feature_space``
(project those features to 2D and colour by class), but:

  * generalised from two conditions to any number of classes, and
  * projections routed through :mod:`mitll_tda.projections`, so PCA, UMAP,
    t-SNE and PHATE are all available (PHATE recommended for latent-space work).
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

from .colors import class_colors, class_markers
from .diagrams import select_dim
from .projections import project, RECOMMENDED_METHOD

__all__ = [
    "topological_features",
    "feature_names",
    "feature_matrix",
    "plot_topological_feature_space",
]

# Per-dimension summary statistics extracted from a diagram.
_STAT_NAMES = [
    "n_features", "mean_life", "max_life", "std_life", "total_persistence",
    "mean_birth", "std_birth", "mean_death", "std_death",
]


def _dim_stats(dgm: np.ndarray) -> np.ndarray:
    """Summary stats for one (m, 2) diagram (finite features only)."""
    d = np.asarray(dgm, dtype="float64").reshape(-1, 2)
    if len(d):
        d = d[np.isfinite(d).all(axis=1)]
    if len(d) == 0:
        return np.zeros(len(_STAT_NAMES))
    life = d[:, 1] - d[:, 0]
    return np.array([
        len(d), life.mean(), life.max(), life.std(), life.sum(),
        d[:, 0].mean(), d[:, 0].std(), d[:, 1].mean(), d[:, 1].std(),
    ])


def topological_features(dgms, dims: Sequence[int] = (0, 1)) -> np.ndarray:
    """Flatten per-dimension summary statistics into one feature vector.

    ``dgms`` is a persim-style list; stats are concatenated across ``dims``.
    """
    return np.concatenate([_dim_stats(select_dim(dgms, d)) for d in dims])


def feature_names(dims: Sequence[int] = (0, 1)) -> List[str]:
    return [f"H{d}_{s}" for d in dims for s in _STAT_NAMES]


def feature_matrix(sample_diagrams: Sequence, labels: Sequence,
                   dims: Sequence[int] = (0, 1)):
    """Stack per-sample topological feature vectors into ``(X, labels, names)``.

    Parameters
    ----------
    sample_diagrams : sequence
        One persim-style diagram list per sample.
    labels : sequence
        Class label per sample (any hashable; any number of classes).
    dims : sequence of int
        Homology dimensions to summarise.
    """
    X = np.vstack([topological_features(d, dims=dims) for d in sample_diagrams])
    return X, np.asarray(labels), feature_names(dims)


def plot_topological_feature_space(
    sample_diagrams: Sequence, labels: Sequence, method: str = RECOMMENDED_METHOD,
    dims: Sequence[int] = (0, 1), ax=None, class_names: Optional[dict] = None,
    standardize: bool = True, title: Optional[str] = None, metric: str = "euclidean",
    **proj_kwargs,
):
    """Project per-sample topological features to 2D and scatter by class.

    Any number of classes is supported; each gets a distinct colour+marker.

    Parameters
    ----------
    sample_diagrams : sequence
        One persim-style diagram list per sample.
    labels : sequence
        Class label per sample.
    method : str
        Projection method ('pca', 'mds', 'tsne', 'umap', 'phate').
    standardize : bool
        Z-score the features before projecting (recommended; the raw stats live
        on very different scales).
    """
    import matplotlib.pyplot as plt

    X, labels_arr, names = feature_matrix(sample_diagrams, labels, dims=dims)

    if standardize:
        mu = X.mean(axis=0)
        sd = X.std(axis=0)
        sd[sd == 0] = 1.0
        X = (X - mu) / sd

    coords = np.asarray(
        project(X, method=method, n_components=2, metric=metric, **proj_kwargs)
    )

    if ax is None:
        _, ax = plt.subplots(figsize=(7, 6))

    colors = class_colors(labels_arr)
    markers = class_markers(labels_arr)
    for lab in sorted(set(labels_arr.tolist()), key=lambda x: (x == -1, x)):
        m = labels_arr == lab
        name = class_names[lab] if class_names and lab in class_names else str(lab)
        ax.scatter(coords[m, 0], coords[m, 1], s=40, color=colors[lab],
                   marker=markers.get(lab, "o"), edgecolors="white", linewidths=0.5,
                   label=name, alpha=0.85)
    ax.set_title(title or f"Topological feature space ({method.upper()})")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    return ax, coords
