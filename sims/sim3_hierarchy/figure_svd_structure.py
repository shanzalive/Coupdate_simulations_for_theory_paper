"""Reproduce Saxe, McClelland & Ganguli 2019 PNAS Fig. 4 B-D, but computed
from OUR own simulated data (the actual sampled branching-diffusion
features, Y_items) rather than their illustrative cartoon. Validates that
our task construction produces the same qualitative structure they report:
(B) a hierarchical item-item correlation matrix (closer tree relatives
more correlated), (C) an SVD whose item-loadings ("semantic distinctions")
mirror the tree topology, (D) singular values ordered by hierarchy level
(coarser splits = larger singular value). Panel (A), their tree diagram,
is schematic (not data-derived) and is skipped here.

Item names and semantic-distinction names are Saxe et al.'s own (Fig. 4A's
tree: Plant{Flower{Daisy,Rose}, Tree{Oak,Pine}}, Animal{Bird{Canary,Robin},
Fish{Salmon,Sunfish}}) -- identical tree topology/leaf order to ours
(train.tree_distance), so this is a direct relabeling, not an approximation.
We additionally keep the "Overall Mean"/baseline distinction and all 4
leaf-level splits (Daisy-Rose, Oak-Pine, Canary-Robin, Salmon-Sunfish),
where their Fig. 4D's cartoon only labeled 3 of the 4 (likely a space
constraint in their illustration, not a mathematical omission).
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np

from sims.sim1_layerwise_coupdate.figure1 import LABEL_FS, TICK_FS, TITLE_FS
from sims.sim3_hierarchy import analysis as A
from sims.sim3_hierarchy.train import haar_tree_wavelets

DEFAULT_OUT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "sim3_hierarchy", "figures", "figure_svd_structure.pdf")

ITEM_NAMES = ["Daisy", "Rose", "Oak", "Pine", "Canary", "Robin", "Salmon", "Sunfish"]
DISTINCTION_NAMES = {
    (0, 0): "Plant-Animal",
    (1, 0): "Flower-Tree", (1, 1): "Bird-Fish",
    (2, 0): "Daisy-Rose", (2, 1): "Oak-Pine", (2, 2): "Canary-Robin", (2, 3): "Salmon-Sunfish",
}
# Shared mode-identity color scheme (figure4.py imports these too, so the two
# figures stay in sync) -- baseline/Plant-Animal/Flower-Tree/Bird-Fish are the
# 4 modes tracked individually elsewhere in the paper; the 4 finest (level 2)
# leaf-pair splits aren't tracked individually anywhere else, so they share
# one color rather than inventing 4 more distinct ones.
BASELINE_COLOR = "#888888"
PLANT_ANIMAL_COLOR = "#4C72B0"
FLOWER_TREE_COLOR = "#C44E52"
BIRD_FISH_COLOR = "#E8989B"
LEAF_COLOR = "#000000"
DISTINCTION_COLORS = {
    "Overall Mean": BASELINE_COLOR,
    "Plant-Animal": PLANT_ANIMAL_COLOR,
    "Flower-Tree": FLOWER_TREE_COLOR,
    "Bird-Fish": BIRD_FISH_COLOR,
    "Daisy-Rose": LEAF_COLOR, "Oak-Pine": LEAF_COLOR,
    "Canary-Robin": LEAF_COLOR, "Salmon-Sunfish": LEAF_COLOR,
}


def _ideal_refs_and_names(n_items: int, depth: int):
    """(n_items, n_items) idealized reference directions (constant/baseline
    + all P-1 exact Haar wavelets) and their names, in the same order."""
    wavelets, levels = haar_tree_wavelets(n_items, depth)
    const = np.ones(n_items) / np.sqrt(n_items)
    refs = np.vstack([const[None, :], wavelets])
    names = ["Overall Mean"]
    node_counter = {}
    for lam in levels:
        lam = int(lam)
        node = node_counter.get(lam, 0)
        names.append(DISTINCTION_NAMES[(lam, node)])
        node_counter[lam] = node + 1
    return refs, names


def _match_empirical_to_names(V: np.ndarray, n_items: int, depth: int) -> list:
    """Name each empirical mode via a one-to-one optimal assignment (Hungarian
    algorithm) against the idealized reference directions, not independent
    per-mode nearest-match -- independent matching let two empirical modes
    both claim the same idealized reference (e.g. two different modes both
    matching "Canary-Robin") while another reference (e.g. "Oak-Pine") went
    unclaimed, since within a level the idealized references are much more
    similar to each other than across levels, and finite-sample noise can
    make an empirical mode's nearest match ambiguous between siblings. A
    one-to-one assignment forces a valid permutation instead. Coarser LEVEL
    classification (train.empirical_modes) doesn't have this problem since
    levels are much better separated than individual same-level nodes.
    """
    from scipy.optimize import linear_sum_assignment
    refs, ref_names = _ideal_refs_and_names(n_items, depth)
    sims = np.abs(V @ refs.T)  # (n_modes, n_refs) cosine similarity (both unit norm)
    row_idx, col_idx = linear_sum_assignment(-sims)  # maximize total similarity
    names = [None] * V.shape[0]
    for r, c in zip(row_idx, col_idx):
        names[r] = ref_names[c]
    return names


def plot_panels(fig, axes, results):
    """Draw the 3 SVD-structure panels (correlation, semantic distinctions,
    singular values) into the given 3 axes -- factored out so figure4.py can
    embed these as a row alongside the co-update panels, not just render
    them standalone via make_figure below."""
    n_items = int(results["n_items"])
    depth = int(results["tree_depth"])
    Y = results["Y_items"]  # (n_items, n_features) -- the actual sampled data
    V = results["V"]  # (n_modes, n_items) -- empirical item-loadings
    s = results["singular_values"]

    names = _match_empirical_to_names(V, n_items, depth)
    order = np.argsort(-s)  # descending singular value, matching Fig. 4D's ordering

    # Panel B: item-item correlation matrix (all entries >= 0 by construction --
    # q_k > 0 for every pair, see train.py's module docstring / SI Appendix).
    ax = axes[0]
    Sigma = (Y @ Y.T) / Y.shape[1]
    im = ax.imshow(Sigma, cmap="Reds", vmin=0, vmax=1)
    ax.set_xticks(range(n_items))
    ax.set_xticklabels(ITEM_NAMES, rotation=90, fontsize=TICK_FS - 1)
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(ITEM_NAMES, fontsize=TICK_FS - 1)
    ax.set_xlabel("Item $j$", fontsize=LABEL_FS)
    ax.set_ylabel("Item $i$", fontsize=LABEL_FS)
    ax.set_title("Item-item correlation", fontsize=TITLE_FS)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FS - 1)

    # Panel C: SVD item-loadings ("semantic distinctions"), ordered by
    # descending singular value, each column a right/item singular vector.
    ax = axes[1]
    V_ord = V[order].T  # (n_items, n_modes)
    vmax = np.abs(V_ord).max()
    im = ax.imshow(V_ord, cmap="PiYG", vmin=-vmax, vmax=vmax)
    ax.set_yticks(range(n_items))
    ax.set_yticklabels(ITEM_NAMES, fontsize=TICK_FS - 1)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([names[i] for i in order], rotation=90, fontsize=TICK_FS - 2)
    ax.set_xlabel("Semantic distinction $\\alpha$", fontsize=LABEL_FS)
    ax.set_ylabel("Item $i$", fontsize=LABEL_FS)
    ax.set_title("SVD semantic distinctions", fontsize=TITLE_FS)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(r"item loading $v_\alpha(i)$", fontsize=TICK_FS - 1)
    cbar.ax.tick_params(labelsize=TICK_FS - 1)

    # Panel D: singular values, ordered descending, colored by mode identity
    # (matches figure4.py's row 2/3 colors; the 4 finest leaf-pair splits
    # share one color since they aren't tracked individually elsewhere).
    ax = axes[2]
    ordered_names = [names[i] for i in order]
    colors = [DISTINCTION_COLORS[n] for n in ordered_names]
    ax.scatter(range(len(order)), s[order], c=colors, s=36, zorder=3)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels(ordered_names, rotation=90, fontsize=TICK_FS - 2)
    ax.set_ylabel("Singular value", fontsize=LABEL_FS)
    ax.set_title("Singular values", fontsize=TITLE_FS)
    ax.tick_params(axis="y", labelsize=TICK_FS)
    ax.set_ylim(0, s.max() * 1.15)


def make_figure(path: str = A.DEFAULT_PATH, out_path: str = DEFAULT_OUT_PATH):
    results = A.load_results(path)
    fig, axes = plt.subplots(1, 3, figsize=(10.5, 3.4))
    plot_panels(fig, axes, results)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    make_figure()
