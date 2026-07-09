"""Tests for the N-class mitll_tda toolkit.

The point of the fork is to lift the two-class-only notebook viz into a reusable
functional API that works for ANY number of classes, so these tests drive the
whole flow with THREE classes (not two) and check that every viz schema and the
feature-space projection accept N classes without special-casing.
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from mitll_tda import (  # noqa: E402
    attention_to_distance,
    compute_diagrams,
    diagram_distance_matrix,
    lifespans,
    select_dim,
    class_colors,
    class_markers,
    sequential_cmaps,
    feature_matrix,
    feature_names,
    project,
    plot_persistence_contours,
    plot_persistence_diagrams,
    plot_persistence_kde,
    plot_survival_curves,
    plot_diagram_distance_matrix,
    plot_topological_feature_space,
    available_methods,
)


def _circle_points(n, r, noise, seed):
    rng = np.random.default_rng(seed)
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.stack([r * np.cos(th), r * np.sin(th)], 1) + noise * rng.standard_normal((n, 2))


def _three_class_diagrams():
    """Three classes = three circles of different radius/noise -> distinct H1."""
    from scipy.spatial.distance import squareform, pdist
    diags = {}
    for lab, (r, noise, seed) in enumerate([(1.0, 0.05, 0), (2.0, 0.08, 1), (0.5, 0.03, 2)]):
        D = squareform(pdist(_circle_points(80, r, noise, seed)))
        diags[f"class_{lab}"] = compute_diagrams(D, maxdim=1)
    return diags


def test_attention_to_distance_is_valid():
    rng = np.random.default_rng(0)
    A = rng.random((12, 12))
    D = attention_to_distance(A, as_distance=True)
    assert D.shape == (12, 12)
    np.testing.assert_allclose(D, D.T, atol=1e-12)
    assert np.allclose(np.diag(D), 0.0)
    assert (D >= 0).all()


def test_colors_and_markers_scale_past_two_classes():
    labels = [0, 1, 2, 3, 4]
    c = class_colors(labels)
    m = class_markers(labels)
    assert len(c) == 5 and len(set(m.values())) >= 5
    # two-class convention preserved: class 0 blue-ish, class 1 orange-ish
    c2 = class_colors([0, 1])
    assert c2[0][2] > c2[0][0]  # blue: more blue than red
    assert c2[1][0] > c2[1][2]  # orange: more red than blue


def test_diagram_distance_matrix_is_symmetric_nxn():
    diags = _three_class_diagrams()
    labels, M = diagram_distance_matrix(diags, dim=1, metric="wasserstein")
    assert M.shape == (3, 3)
    np.testing.assert_allclose(M, M.T, atol=1e-9)
    assert np.allclose(np.diag(M), 0.0)


@pytest.mark.parametrize("plot_fn", [
    plot_persistence_contours,
    plot_persistence_kde,
    plot_survival_curves,
])
def test_overlay_viz_accept_three_classes(plot_fn):
    diags = _three_class_diagrams()
    ax = plot_fn(diags, dim=1)
    # every class should appear in the legend
    texts = [t.get_text() for t in ax.get_legend().get_texts()] if ax.get_legend() else []
    assert len(texts) >= 3 or plot_fn is plot_persistence_contours


def test_per_class_diagram_panels():
    diags = _three_class_diagrams()
    axes = plot_persistence_diagrams(diags, dim=1)
    assert len(np.atleast_1d(axes)) == 3


def test_distance_heatmap_runs():
    diags = _three_class_diagrams()
    ax, (labels, M) = plot_diagram_distance_matrix(diags, dim=1, metric="bottleneck")
    assert M.shape == (3, 3)


@pytest.mark.parametrize("method", ["pca", "phate"])
def test_feature_space_projection_three_classes(method):
    if not available_methods().get(method, False):
        pytest.skip(f"{method} not installed")
    # 30 samples across 3 classes, each a noisy circle
    from scipy.spatial.distance import squareform, pdist
    sample_diagrams, labels = [], []
    for lab, (r, noise) in enumerate([(1.0, 0.05), (2.0, 0.08), (0.5, 0.03)]):
        for s in range(10):
            D = squareform(pdist(_circle_points(60, r, noise, seed=100 * lab + s)))
            sample_diagrams.append(compute_diagrams(D, maxdim=1))
            labels.append(lab)
    ax, coords = plot_topological_feature_space(
        sample_diagrams, labels, method=method, dims=(0, 1)
    )
    assert coords.shape == (30, 2)
    assert np.isfinite(coords).all()


# --- edge cases / regressions added after review ----------------------------

def _diags_with_noise():
    """Three classes, one of which is the noise label -1 (with real H1 features)."""
    from scipy.spatial.distance import squareform, pdist
    diags = {}
    for lab, (r, noise, seed) in [(0, (1.0, 0.05, 0)), (1, (2.0, 0.08, 1)), (-1, (0.5, 0.03, 2))]:
        D = squareform(pdist(_circle_points(80, r, noise, seed)))
        diags[lab] = compute_diagrams(D, maxdim=1)
    return diags


def test_contour_overlay_handles_noise_label():
    # regression: the -1 (noise) class used to raise KeyError in the contour
    # overlay because class_markers / sequential_cmaps dropped -1.
    assert plot_persistence_contours(_diags_with_noise(), dim=1) is not None


def test_markers_and_cmaps_include_noise_label():
    for fn in (class_markers, sequential_cmaps):
        assert -1 in fn([0, 1, -1]), f"{fn.__name__} dropped the -1 noise label"


def test_two_class_marker_backcompat():
    m = class_markers([0, 1])
    assert m[0] == "o" and m[1] == "^"   # preserve the paper's circle/triangle


def test_single_class_viz_runs():
    one = dict(list(_three_class_diagrams().items())[:1])
    assert plot_persistence_contours(one, dim=1) is not None
    assert plot_persistence_kde(one, dim=1) is not None
    assert plot_survival_curves(one, dim=1) is not None
    assert len(np.atleast_1d(plot_persistence_diagrams(one, dim=1))) == 1


def test_empty_diagrams_dont_crash():
    empty = {0: [np.empty((0, 2)), np.empty((0, 2))],
             1: [np.empty((0, 2)), np.empty((0, 2))]}
    assert plot_persistence_contours(empty, dim=1) is not None   # -> "No finite features"
    assert plot_persistence_kde(empty, dim=1) is not None
    assert plot_survival_curves(empty, dim=1) is not None


def test_select_dim_and_lifespans():
    dgms = [np.array([[0.0, 1.0]]), np.array([[0.2, 0.9], [0.1, np.inf]])]
    assert select_dim(dgms, 1).shape == (2, 2)
    assert select_dim(dgms, 5).shape == (0, 2)                    # out-of-range dim
    assert select_dim(np.array([[0.0, 1.0]]), 1).shape == (1, 2)  # already (m, 2)
    life = lifespans(dgms, dim=1)                                 # finite only -> drops inf
    assert life.shape == (1,) and np.isclose(life[0], 0.7)


def test_feature_matrix_and_names():
    diags = list(_three_class_diagrams().values())
    X, y, names = feature_matrix(diags, [0, 1, 2], dims=(0, 1))
    assert X.shape == (3, len(names))
    assert list(feature_names((0, 1))) == names
    assert len(names) == 18                                       # 9 stats * 2 dims


def test_compute_diagrams_rejects_asymmetric():
    D = np.array([[0.0, 1.0, 2.0], [0.9, 0.0, 1.0], [2.0, 1.0, 0.0]])  # asymmetric
    with pytest.raises(ValueError):
        compute_diagrams(D, maxdim=1)


def test_attention_to_distance_modes_and_errors():
    A = np.random.default_rng(1).random((6, 6))
    for mode in ("avg", "max", "min"):
        D = attention_to_distance(A, symmetrize=mode)
        np.testing.assert_allclose(D, D.T, atol=1e-12)
        assert np.allclose(np.diag(D), 0.0)
    with pytest.raises(ValueError):
        attention_to_distance(np.zeros((3, 4)))                   # non-square
    with pytest.raises(ValueError):
        attention_to_distance(A, symmetrize="bogus")


def test_project_mds_points_and_precomputed():
    from scipy.spatial.distance import squareform, pdist
    pts = _circle_points(40, 1.0, 0.02, seed=7)
    coords = np.asarray(project(pts, method="mds", n_components=2))
    assert coords.shape == (40, 2) and np.isfinite(coords).all()
    # a distance matrix is auto-detected as precomputed (square+symmetric+zero-diag)
    D = squareform(pdist(pts))
    coords_pc = np.asarray(project(D, method="mds", n_components=2))
    assert coords_pc.shape == (40, 2) and np.isfinite(coords_pc).all()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
