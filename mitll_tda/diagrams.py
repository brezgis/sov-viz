"""Persistence-diagram computation and comparison, decoupled from any model.

The "shape of vulnerability" notebooks compute persistence with the ``ripser``
Python package and compare diagrams pairwise (original vs perturbed / sandbagging
vs non-sandbagging) with ``persim.wasserstein`` / ``persim.bottleneck``.  This
module lifts that logic out of the notebooks and generalises the comparison from
a single pair to an N-class distance MATRIX, so any number of conditions can be
compared at once.

Persistence is computed from a distance matrix.  ``ripser`` is used when it is
installed (matching the notebooks); otherwise we fall back to ``gudhi`` (already
a dependency of the sibling TDA tools), so nothing hard-requires ``ripser``.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Union

import numpy as np

__all__ = [
    "attention_to_distance",
    "compute_diagrams",
    "lifespans",
    "select_dim",
    "diagram_distance_matrix",
]


def attention_to_distance(attention: np.ndarray, symmetrize: str = "avg",
                          as_distance: bool = True) -> np.ndarray:
    """Turn a (seq, seq) attention matrix into a symmetric distance matrix.

    Reconciles the two conventions that had drifted between the notebook
    families: ``analyze_pairs`` used the attention as a distance directly, while
    the sandbagging notebooks used ``1 - similarity``.  ``as_distance=True``
    (default) selects the ``1 - S`` convention (higher attention -> smaller
    distance), which is the sensible one for persistent homology; pass
    ``as_distance=False`` to treat the (symmetrised) attention itself as the
    dissimilarity.

    Parameters
    ----------
    attention : np.ndarray
        ``(seq, seq)`` attention weights (already averaged over heads, say).
    symmetrize : {"avg", "max", "min"}
        How to symmetrise the (generally non-symmetric) attention.
    as_distance : bool
        If True, return ``1 - S_normalised``; else return the symmetrised
        attention as the dissimilarity. The diagonal is always zeroed.
    """
    A = np.asarray(attention, dtype="float64")
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError(f"attention must be square (seq, seq); got {A.shape}")

    if symmetrize == "avg":
        S = 0.5 * (A + A.T)
    elif symmetrize == "max":
        S = np.maximum(A, A.T)
    elif symmetrize == "min":
        S = np.minimum(A, A.T)
    else:
        raise ValueError("symmetrize must be 'avg', 'max' or 'min'")

    if as_distance:
        smax = S.max()
        S = S / smax if smax > 0 else S
        D = 1.0 - S
    else:
        D = S
    np.fill_diagonal(D, 0.0)
    # guard against tiny negative/asymmetry from float ops
    D = 0.5 * (D + D.T)
    np.clip(D, 0.0, None, out=D)
    np.fill_diagonal(D, 0.0)
    return D


def compute_diagrams(distance_matrix: np.ndarray, maxdim: int = 1,
                     thresh: Optional[float] = None) -> List[np.ndarray]:
    """Compute persistence diagrams from a distance matrix.

    Prefers the ``ripser`` package (as in the original notebooks); falls back to
    ``gudhi`` when ripser is not installed.

    Returns a persim-style list ``[H0, H1, ...]`` where each entry is an
    ``(m, 2)`` array of [birth, death] pairs (infinite deaths kept as ``inf``).
    """
    D = np.asarray(distance_matrix, dtype="float64")
    if D.ndim != 2 or D.shape[0] != D.shape[1]:
        raise ValueError(f"distance_matrix must be square (n, n); got shape {D.shape}")
    if D.size and not np.allclose(D, D.T):
        raise ValueError(
            "distance_matrix must be symmetric; got an asymmetric matrix "
            "(ripser/gudhi would silently use only the upper triangle). "
            "Symmetrise it first, e.g. D = 0.5 * (D + D.T)."
        )

    try:
        from ripser import ripser as _ripser

        kw = {"distance_matrix": True, "maxdim": maxdim}
        if thresh is not None:
            kw["thresh"] = thresh
        return [np.asarray(d, dtype="float64") for d in _ripser(D, **kw)["dgms"]]
    except ImportError:
        pass

    import gudhi as gd

    max_edge = thresh if thresh is not None else float("inf")
    rc = gd.RipsComplex(distance_matrix=D, max_edge_length=max_edge)
    st = rc.create_simplex_tree(max_dimension=maxdim + 1)
    st.compute_persistence()
    dgms = []
    for d in range(maxdim + 1):
        arr = np.asarray(st.persistence_intervals_in_dimension(d), dtype="float64")
        dgms.append(arr.reshape(-1, 2) if arr.size else np.empty((0, 2)))
    return dgms


def select_dim(dgms, dim: int = 1) -> np.ndarray:
    """Pull out the dimension-``dim`` diagram from either a persim-style list or
    an already-selected ``(m, 2)`` array (returned unchanged)."""
    # A persim-style diagram list holds per-dimension arrays with *different*
    # row counts, so it must be indexed as a sequence -- never fed whole to
    # np.asarray (which raises on the ragged shape).
    if isinstance(dgms, (list, tuple)):
        if dim < len(dgms):
            return np.asarray(dgms[dim], dtype="float64").reshape(-1, 2)
        return np.empty((0, 2))
    arr = np.asarray(dgms, dtype="float64")
    if arr.dtype == object:  # object-array of per-dim diagrams
        if dim < len(arr):
            return np.asarray(arr[dim], dtype="float64").reshape(-1, 2)
        return np.empty((0, 2))
    if arr.ndim == 2 and arr.shape[1] == 2:  # already a single (m, 2) diagram
        return arr
    if arr.size == 0:
        return np.empty((0, 2))
    raise ValueError("expected a persim-style list of diagrams or an (m, 2) array")


def lifespans(dgm, dim: Optional[int] = None, finite_only: bool = True) -> np.ndarray:
    """Persistence lifespans (death - birth) for a diagram.

    ``dim`` may be given to first select a dimension from a persim-style list.
    """
    d = select_dim(dgm, dim) if dim is not None else np.asarray(dgm, dtype="float64").reshape(-1, 2)
    if len(d) == 0:
        return np.empty((0,))
    life = d[:, 1] - d[:, 0]
    if finite_only:
        life = life[np.isfinite(life)]
    return life


def diagram_distance_matrix(
    class_diagrams: Dict, dim: int = 1, metric: str = "wasserstein",
) -> "tuple[list, np.ndarray]":
    """Pairwise distances between per-class diagrams as an N x N matrix.

    Generalises the notebooks' single original-vs-perturbed ``wasserstein`` call
    to any number of classes: every pair of conditions is compared, yielding a
    symmetric ``(N, N)`` matrix ready for a heatmap / MDS / clustering.

    Parameters
    ----------
    class_diagrams : dict
        ``{label: diagram}`` where ``diagram`` is a persim-style list or an
        ``(m, 2)`` array. The dimension ``dim`` is selected from lists.
    dim : int
        Homology dimension to compare (default H1).
    metric : {"wasserstein", "bottleneck"}
        Diagram distance (via ``persim``).

    Returns
    -------
    (labels, matrix) : (list, np.ndarray)
        ``labels`` are the class labels in matrix order; ``matrix[i, j]`` is the
        distance between class ``labels[i]`` and ``labels[j]``.
    """
    from persim import wasserstein, bottleneck

    dist_fn = {"wasserstein": wasserstein, "bottleneck": bottleneck}.get(metric)
    if dist_fn is None:
        raise ValueError("metric must be 'wasserstein' or 'bottleneck'")

    labels = list(class_diagrams.keys())
    diags = {lab: select_dim(class_diagrams[lab], dim) for lab in labels}
    n = len(labels)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(dist_fn(diags[labels[i]], diags[labels[j]]))
            M[i, j] = M[j, i] = d
    return labels, M
