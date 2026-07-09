# Setup with uv

This fork adds a reusable library package, `mitll_tda`, with a PEP 621
`pyproject.toml`. [uv](https://docs.astral.sh/uv/) manages it directly:

```bash
uv venv                                   # Python >=3.9
source .venv/bin/activate
uv pip install -e '.[projections]'        # mitll_tda + PCA/UMAP/t-SNE/PHATE
```

Extras:

| extra          | what it adds                                                       |
|----------------|-------------------------------------------------------------------|
| `projections`  | UMAP + PHATE + openTSNE for the topological feature space         |
| `ripser`       | the `ripser` PH engine (gudhi is the built-in fallback)           |
| `pipeline`     | torch + transformers for the text→attention→diagram pipeline       |
| `dev`          | pytest                                                             |

The original analysis notebooks (repo root and `mitll_adam/`) are unchanged;
`mitll_tda` is the reusable, **N-class** distillation of their viz schemas.

## N-class functional API

```python
from mitll_tda import (compute_diagrams, attention_to_distance,
                       plot_persistence_contours, plot_diagram_distance_matrix,
                       plot_topological_feature_space)

# any number of classes, keyed by arbitrary labels
diags = {cls: compute_diagrams(attention_to_distance(attn[cls]), maxdim=1)
         for cls in classes}
plot_persistence_contours(diags, dim=1)               # overlaid, N classes
plot_diagram_distance_matrix(diags, metric="wasserstein")   # N x N heatmap
```

## Quick check

```bash
uv run python -m pytest mitll_tda/tests/ -q
```
