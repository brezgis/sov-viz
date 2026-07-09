# sov-viz

A fork of [**angelinatsai04/mitll_clinic**](https://github.com/angelinatsai04/mitll_clinic) that **expands and adapts the authors' topological visualization suite** from their paper:

> **The Shape of Vulnerability: How Adversarial Perturbations Reshape the Topology of Language Model Latent Spaces**
> Angelina Tsai, Shreya Subramanian, Catherine Liu, Kimberly Lopez, Leif Zinn-Brooks, Alexia E. Schulz, Adaku Uchendu — ACL 2026 (SRW)
> 📄 https://aclanthology.org/2026.acl-srw.24/

The paper's "shape of vulnerability" figures compare two conditions (clean vs adversarial) via persistent homology of transformer attention. This fork lifts those visualizations out of the notebooks into a small functional API (`mitll_tda`) and generalizes them from two classes to **any number of classes**. See the paper for the method and results — this repo is the visualization toolkit. The original notebooks are preserved alongside it.

## The visualizations

| Function | Figure |
|---|---|
| `plot_persistence_contours` | contour overlay — per-class KDE of (birth, death) points |
| `plot_persistence_diagrams` | per-class persistence-diagram panels |
| `plot_persistence_kde` / `plot_survival_curves` | lifespan density / survival curves |
| `plot_diagram_distance_matrix` | N×N Wasserstein / bottleneck distance between classes |
| `plot_topological_feature_space` | per-sample topological features projected (PCA/UMAP/t-SNE/PHATE) |

## Install

```
pip install -e .                 # core (matplotlib, gudhi, persim, seaborn, …)
pip install -e '.[projections]'  # + UMAP / PHATE / openTSNE projection backends
```

## Quick use

```python
from mitll_tda import compute_diagrams, attention_to_distance, plot_persistence_contours

# per-class persistence diagrams keyed by any label
diags = {cls: compute_diagrams(attention_to_distance(attn[cls]), maxdim=1)
         for cls in classes}
plot_persistence_contours(diags, dim=1)   # overlaid contour comparison, N classes
```

See `SETUP.md` for the uv-based setup.
