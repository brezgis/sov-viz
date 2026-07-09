"""mitll_tda -- reusable, N-class topological-analysis toolkit.

A functional-API distillation of the "shape of vulnerability" notebooks
(https://github.com/angelinatsai04/mitll_clinic). The notebook code compares
exactly two conditions (clean vs adversarial / sandbagging vs not) with a
hard-coded blue/orange, circle/triangle scheme; this package lifts the reusable
pieces out of the notebooks and generalises them to ANY number of classes, and
adds PCA/UMAP/t-SNE/PHATE projections for the topological feature space.

Modules
-------
- ``diagrams``    : compute persistence from a distance matrix (ripser or gudhi),
                    lifespans, and N x N inter-class diagram-distance matrices.
- ``viz``         : N-class persistence viz -- contour overlay, per-class
                    diagrams, lifespan KDE, survival curves, distance heatmap.
- ``features``    : per-sample topological summary features + feature-space
                    projection (PCA/UMAP/t-SNE/PHATE), coloured by class.
- ``colors``      : colour / marker / colormap cycles that scale past two classes
                    (class 0 = blue/o, class 1 = orange/^, preserving the paper).
- ``projections`` : the shared low-dimensional projection module.

Typical use::

    from mitll_tda import compute_diagrams, attention_to_distance
    from mitll_tda import plot_persistence_contours, plot_diagram_distance_matrix

    # per-class diagrams keyed by an arbitrary label
    diags = {cls: compute_diagrams(attention_to_distance(attn[cls]), maxdim=1)
             for cls in classes}
    plot_persistence_contours(diags, dim=1)          # overlaid, N classes
    plot_diagram_distance_matrix(diags, metric="wasserstein")
"""

from .diagrams import (
    attention_to_distance,
    compute_diagrams,
    lifespans,
    select_dim,
    diagram_distance_matrix,
)
from .viz import (
    plot_persistence_contours,
    plot_persistence_diagrams,
    plot_persistence_kde,
    plot_survival_curves,
    plot_diagram_distance_matrix,
)
from .features import (
    topological_features,
    feature_names,
    feature_matrix,
    plot_topological_feature_space,
)
from .colors import class_colors, class_markers, class_style, sequential_cmaps
from .projections import project, available_methods, METHODS, RECOMMENDED_METHOD

__version__ = "0.1.0"

__all__ = [
    # diagrams
    "attention_to_distance",
    "compute_diagrams",
    "lifespans",
    "select_dim",
    "diagram_distance_matrix",
    # viz
    "plot_persistence_contours",
    "plot_persistence_diagrams",
    "plot_persistence_kde",
    "plot_survival_curves",
    "plot_diagram_distance_matrix",
    # features
    "topological_features",
    "feature_names",
    "feature_matrix",
    "plot_topological_feature_space",
    # colours
    "class_colors",
    "class_markers",
    "class_style",
    "sequential_cmaps",
    # projections
    "project",
    "available_methods",
    "METHODS",
    "RECOMMENDED_METHOD",
]
