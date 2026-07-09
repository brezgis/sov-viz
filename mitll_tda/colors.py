"""Colour and marker cycles for an arbitrary number of classes.

The original "shape of vulnerability" notebooks hard-code a two-class scheme --
blue circles for the attacked/perturbed/sandbagging class, orange triangles for
the clean/original one -- with literals like
``[('Sandbagging','blue','o'), ('Non-Sandbagging','orange','^')]`` repeated in
every plotting function.  That caps every visualisation at two classes.

This module replaces those literals with cycles that scale to any number of
classes while preserving the original two-class look (class 0 = blue/``o``,
class 1 = orange/``^``) so existing figures are reproduced when N == 2.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

__all__ = [
    "class_colors",
    "class_markers",
    "class_style",
    "sequential_cmaps",
]

# Preserve the paper's two-class identity, then extend with distinct hues.
# Index 0 -> blue (attacked/perturbed), 1 -> orange (clean/original).
_BASE_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
    "#e377c2",  # pink
    "#7f7f7f",  # grey
    "#bcbd22",  # olive
    "#17becf",  # cyan
]

_BASE_MARKERS = ["o", "^", "s", "D", "v", "P", "X", "*", "<", ">"]

# Sequential colormaps for per-class KDE/contour fills, matching the original
# Blues (class 0) / Oranges (class 1) choice, then extending.
_BASE_CMAPS = [
    "Blues", "Oranges", "Greens", "Reds", "Purples",
    "YlOrBr", "PuRd", "Greys", "YlGn", "GnBu",
]


def _labels_in_order(class_labels) -> list:
    """Distinct labels in a stable order (sorted; -1/noise pushed to the end)."""
    labels = sorted(set(np.asarray(class_labels).tolist()), key=lambda x: (x == -1, x))
    return labels


def class_colors(class_labels) -> dict:
    """Map each distinct label to an RGBA colour, for any number of classes.

    Uses the 10-colour tab-style base palette (blue, orange, ... preserving the
    two-class convention), then evenly spaced HSV hues beyond 10 classes. Label
    ``-1`` (noise/unassigned) is always grey.
    """
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt

    labels = _labels_in_order(class_labels)
    non_noise = [l for l in labels if l != -1]
    n = len(non_noise)
    out = {}
    for i, label in enumerate(non_noise):
        if n <= len(_BASE_COLORS):
            out[label] = mcolors.to_rgba(_BASE_COLORS[i])
        else:
            out[label] = plt.cm.hsv(i / max(1, n))
    if -1 in labels:
        out[-1] = (0.5, 0.5, 0.5, 1.0)
    return out


def class_markers(class_labels) -> dict:
    """Map each distinct label to a matplotlib marker, cycling for many classes.

    Label ``-1`` (noise/unassigned) is always ``"x"``, matching the grey colour
    :func:`class_colors` assigns it.
    """
    labels = _labels_in_order(class_labels)
    non_noise = [l for l in labels if l != -1]
    out = {label: _BASE_MARKERS[i % len(_BASE_MARKERS)] for i, label in enumerate(non_noise)}
    if -1 in labels:
        out[-1] = "x"
    return out


def sequential_cmaps(class_labels) -> dict:
    """Map each distinct label to a sequential colormap name (for KDE fills).

    Label ``-1`` (noise/unassigned) is always ``"Greys"``.
    """
    labels = _labels_in_order(class_labels)
    non_noise = [l for l in labels if l != -1]
    out = {label: _BASE_CMAPS[i % len(_BASE_CMAPS)] for i, label in enumerate(non_noise)}
    if -1 in labels:
        out[-1] = "Greys"
    return out


def class_style(class_labels) -> dict:
    """Convenience: ``{label: {"color", "marker", "cmap"}}`` for all labels."""
    colors = class_colors(class_labels)
    markers = class_markers(class_labels)
    cmaps = sequential_cmaps(class_labels)
    return {
        label: {
            "color": colors[label],
            "marker": markers.get(label, "o"),
            "cmap": cmaps.get(label, "Greys"),
        }
        for label in colors
    }
