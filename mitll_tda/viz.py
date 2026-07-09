"""N-class persistence-diagram visualisations.

These are the "shape of vulnerability" viz schemas -- contour overlay, per-class
persistence diagrams, lifespan KDE, survival curves, and diagram-distance
heatmaps -- lifted out of the notebooks and generalised from the hard-coded two
classes (blue "perturbed" vs orange "original") to ANY number of classes.

Every function takes ``class_diagrams``: a ``dict`` mapping a class label to that
class's persistence diagram (either a persim-style ``[H0, H1, ...]`` list or an
already-selected ``(m, 2)`` array; the ``dim`` argument selects from lists).
Colours, markers and sequential colormaps are assigned per class by
:mod:`mitll_tda.colors`, which preserves the original blue/orange, circle/
triangle look when there are exactly two classes.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np

from .colors import class_colors, class_markers, sequential_cmaps
from .diagrams import select_dim, lifespans, diagram_distance_matrix

__all__ = [
    "plot_persistence_contours",
    "plot_persistence_diagrams",
    "plot_persistence_kde",
    "plot_survival_curves",
    "plot_diagram_distance_matrix",
]


def _display_name(label, class_names):
    if class_names and label in class_names:
        return class_names[label]
    return str(label)


def _kde_grid(points: np.ndarray, xlim, ylim, gridsize: int = 100, bw=0.1):
    """2D Gaussian KDE over a birth/death point set, evaluated on a grid."""
    from scipy.stats import gaussian_kde

    if len(points) < 3:
        return None
    xx, yy = np.mgrid[
        xlim[0]:xlim[1]:complex(gridsize),
        ylim[0]:ylim[1]:complex(gridsize),
    ]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    try:
        kde = gaussian_kde(points.T, bw_method=bw)
    except np.linalg.LinAlgError:
        return None  # singular (e.g. all points collinear)
    z = kde(positions).reshape(xx.shape)
    return xx, yy, z


def plot_persistence_contours(
    class_diagrams: Dict, dim: int = 1, ax=None, class_names: Optional[dict] = None,
    gridsize: int = 100, levels: int = 8, bw=0.1, title: Optional[str] = None,
    scatter: bool = True,
):
    """Overlaid 2D-KDE contours of the (birth, death) points, one field per class.

    Generalises the notebooks' Blues/Oranges "contour overlay" to N classes,
    each drawn with its own sequential colormap and marker.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 7))

    labels = list(class_diagrams.keys())
    diags = {lab: select_dim(class_diagrams[lab], dim) for lab in labels}
    colors = class_colors(labels)
    cmaps = sequential_cmaps(labels)
    markers = class_markers(labels)

    # common birth/death extent (finite points only), padded a little
    pts_all = np.vstack([d[np.isfinite(d).all(axis=1)] for d in diags.values() if len(d)]) \
        if any(len(d) for d in diags.values()) else np.empty((0, 2))
    if len(pts_all) == 0:
        ax.text(0.5, 0.5, "No finite features", ha="center", va="center",
                transform=ax.transAxes)
        return ax
    lo = pts_all.min(axis=0)
    hi = pts_all.max(axis=0)
    pad = 0.05 * (hi - lo + 1e-9)
    xlim = (lo[0] - pad[0], hi[0] + pad[0])
    ylim = (lo[1] - pad[1], hi[1] + pad[1])

    for lab in labels:
        d = diags[lab]
        d = d[np.isfinite(d).all(axis=1)]
        if len(d) == 0:
            continue
        grid = _kde_grid(d, xlim, ylim, gridsize=gridsize, bw=bw)
        name = _display_name(lab, class_names)
        if grid is not None:
            xx, yy, z = grid
            ax.contour(xx, yy, z, levels=levels, cmap=cmaps[lab],
                       alpha=0.9, linewidths=1.5)
        if scatter:
            ax.scatter(d[:, 0], d[:, 1], s=22, color=colors[lab],
                       marker=markers[lab], edgecolors="white", linewidths=0.4,
                       label=f"{name} points", zorder=3)

    # birth = death diagonal
    dmax = max(xlim[1], ylim[1])
    ax.plot([0, dmax], [0, dmax], "k--", lw=1, alpha=0.6)
    ax.set_xlabel("Birth")
    ax.set_ylabel("Death")
    ax.set_title(title or f"$H_{{{dim}}}$ persistence contour overlay")
    if scatter:
        ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    return ax


def plot_persistence_diagrams(
    class_diagrams: Dict, dim: int = 1, axes=None, class_names: Optional[dict] = None,
    title: Optional[str] = None, shared_range: bool = True,
):
    """One persistence-diagram panel per class (a row of scatter plots).

    Replaces the notebooks' fixed 1x2 original/perturbed layout with a 1xN row.
    """
    import matplotlib.pyplot as plt

    labels = list(class_diagrams.keys())
    diags = {lab: select_dim(class_diagrams[lab], dim) for lab in labels}
    colors = class_colors(labels)
    n = len(labels)

    if axes is None:
        _, axes = plt.subplots(1, n, figsize=(4 * n, 4), squeeze=False)
        axes = axes[0]
    axes = np.atleast_1d(axes)

    finite = [v for d in diags.values() if len(d)
              for v in d[np.isfinite(d).all(axis=1)].ravel()]
    vmax = max(finite) if finite else 1.0

    for ax, lab in zip(axes, labels):
        d = diags[lab]
        name = _display_name(lab, class_names)
        if len(d):
            df = d.copy()
            df[~np.isfinite(df)] = vmax * 1.05
            ax.scatter(df[:, 0], df[:, 1], s=24, color=colors[lab],
                       edgecolors="white", linewidths=0.4)
        ax.plot([0, vmax], [0, vmax], "k--", lw=1, alpha=0.6)
        if shared_range:
            ax.set_xlim(-0.02 * vmax, vmax * 1.1)
            ax.set_ylim(-0.02 * vmax, vmax * 1.1)
        ax.set_title(name)
        ax.set_xlabel("Birth")
        ax.set_ylabel("Death")
    if title:
        axes[0].figure.suptitle(title)
    return axes


def plot_persistence_kde(
    class_diagrams: Dict, dim: int = 1, ax=None, class_names: Optional[dict] = None,
    title: Optional[str] = None, bw_adjust: float = 1.0,
):
    """Overlaid KDE of persistence lifespans (death - birth), one curve per class."""
    import matplotlib.pyplot as plt

    try:
        import seaborn as sns
        _has_sns = True
    except ImportError:
        _has_sns = False

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    labels = list(class_diagrams.keys())
    colors = class_colors(labels)
    for lab in labels:
        life = lifespans(class_diagrams[lab], dim=dim)
        if len(life) < 2:
            continue
        name = _display_name(lab, class_names)
        if _has_sns:
            sns.kdeplot(life, ax=ax, label=name, color=colors[lab],
                        bw_adjust=bw_adjust, fill=False)
        else:
            # simple histogram-density fallback
            ax.hist(life, bins=30, density=True, histtype="step",
                    color=colors[lab], label=name)
    ax.set_xlabel("Persistence")
    ax.set_ylabel("Density")
    ax.set_title(title or f"$H_{{{dim}}}$ persistence KDE")
    ax.legend()
    return ax


def plot_survival_curves(
    class_diagrams: Dict, dim: int = 1, ax=None, class_names: Optional[dict] = None,
    title: Optional[str] = None, logy: bool = True,
):
    """Survival curves P(persistence > t), one per class (log-y by default)."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    labels = list(class_diagrams.keys())
    colors = class_colors(labels)
    for lab in labels:
        life = np.sort(lifespans(class_diagrams[lab], dim=dim))
        if len(life) == 0:
            continue
        y = 1.0 - np.arange(len(life)) / len(life)
        name = _display_name(lab, class_names)
        ax.plot(life, y, label=name, color=colors[lab], lw=2)
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel("Persistence")
    ax.set_ylabel(r"$P(\mathrm{persistence} > t)$")
    ax.set_title(title or f"$H_{{{dim}}}$ persistence survival curve")
    ax.legend()
    return ax


def plot_diagram_distance_matrix(
    class_diagrams: Dict, dim: int = 1, metric: str = "wasserstein", ax=None,
    class_names: Optional[dict] = None, title: Optional[str] = None, annot: bool = True,
):
    """Heatmap of the N x N pairwise diagram distances between classes.

    Generalises the notebooks' single original-vs-perturbed Wasserstein number to
    a full inter-class distance matrix.
    """
    import matplotlib.pyplot as plt

    labels, M = diagram_distance_matrix(class_diagrams, dim=dim, metric=metric)
    names = [_display_name(l, class_names) for l in labels]

    if ax is None:
        _, ax = plt.subplots(figsize=(1.4 * len(labels) + 2, 1.2 * len(labels) + 2))

    im = ax.imshow(M, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(names, rotation=45, ha="right")
    ax.set_yticklabels(names)
    if annot:
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{M[i, j]:.2g}", ha="center", va="center",
                        color="white" if M[i, j] < M.max() / 2 else "black",
                        fontsize=8)
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                       label=f"{metric} distance")
    ax.set_title(title or f"$H_{{{dim}}}$ {metric} distance between classes")
    return ax, (labels, M)
