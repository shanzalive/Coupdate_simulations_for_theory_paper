"""Camera-ready paper figure: coarse-to-fine co-update waves in a binary
hierarchical task.

Layout (4 stacked rows):
  Row 1 (3 columns): the SVD-structure panels (figure_svd_structure.py's
    correlation matrix / semantic distinctions / singular values), embedded
    here as grounding context for the co-update rows below.
  Row 2 (1 column, twin axis): ALL modes' raw amplitude a_alpha(t) (solid)
    and anchor-centered co-update profile a_alpha(t)*(s_alpha-a_alpha(t))
    (dashed) overlaid on one plot. Baseline=grey, Plant-Animal=blue, and
    level 1's two modes (Flower-Tree, Bird-Fish) in two shades of the same
    red family, since they're both level 1. Two legends: one for which
    mode (color -> semantic distinction), one for which quantity
    (linestyle -> amplitude vs. co-update profile).
  Row 3 (4 columns): item-item co-update matrix C^(0)_2(i,j) =
    <Delta_0 h2(i), Delta_0 h2(j)> at 4 snapshots -- each mode's OWN peak
    trial (Baseline, Plant-Animal, Flower-Tree, Bird-Fish shown
    separately, not combined into per-level panels): positive means items
    i and j's representations shifted in the SAME direction when anchor 0
    was trained (they co-updated together); negative means opposite
    directions. Baseline's own heatmap is shown RAW (its point is to show
    what the near-uniform baseline mode itself looks like); the others are
    mean-centered to remove baseline's near-constant additive offset,
    which otherwise washes out their sign-alternating tree-split
    structure (baseline is kept in the task, not filtered -- matches Saxe
    et al.'s own Fig. 4D, where "Overall Mean" is also the largest
    singular value -- see train.py's module docstring). Bird-Fish's panel
    is expected to show ~nothing: it has ZERO loading at anchor=Daisy, so
    it contributes EXACTLY zero to Delta_0 h2 for any item by construction
    (every term in C_analytical/b_norm_sq_analytical is multiplied by
    V[mode,anchor]) -- shown deliberately, as a direct visual contrast to
    Flower-Tree (which DOES involve Daisy and shows real structure).
  Row 4 (1 column): per-probe co-update trajectory over the whole run
    (||Delta_0 h2(probe)|| vs. trial, at every trial anchor 0 was actually
    trained), colored by tree distance to the anchor (1=closest ..
    depth=farthest).

Item ticks on the heatmaps use Saxe et al.'s own Fig. 4A names (Daisy=
item 0/anchor, Rose, Oak, Pine, Canary, Robin, Salmon, Sunfish) -- their
tree topology and leaf order are identical to ours (train.tree_distance),
so this is a direct, unambiguous relabeling, not an approximation.

Panel letters intentionally omitted -- added later in Illustrator.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from sims.sim1_layerwise_coupdate.figure1 import LABEL_FS, TICK_FS, TITLE_FS
from sims.sim3_hierarchy import analysis as A
from sims.sim3_hierarchy import figure_svd_structure as SVD
from sims.sim3_hierarchy.train import tree_distance

DEFAULT_OUT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "sim3_hierarchy", "figures", "figure4_hierarchy.pdf")

ANCHOR = A.ANCHOR
# Mode-identity colors are defined once in figure_svd_structure.py (shared with
# its Panel D) and reused here so both figures agree.
BASELINE_COLOR = SVD.BASELINE_COLOR
PLANT_ANIMAL_COLOR = SVD.PLANT_ANIMAL_COLOR
FLOWER_TREE_COLOR = SVD.FLOWER_TREE_COLOR  # level 1's anchor-relevant mode (involves Daisy)
BIRD_FISH_COLOR = SVD.BIRD_FISH_COLOR      # level 1's OTHER mode (doesn't involve Daisy)
ITEM_NAMES = SVD.ITEM_NAMES
# Sequential (not mode-colored) palette for the bottom panel: keyed by tree
# DISTANCE to the anchor (1=closest possible distinct item .. tree_depth=
# farthest), dark->light = close->far.
DIST_COLORS = {1: "#08519c", 2: "#6baed6", 3: "#c6dbef"}


def _probe_order(results, anchor: int):
    n_items = int(results["n_items"])
    depth = int(results["tree_depth"])
    probes = list(range(n_items))
    probes.remove(anchor)
    dists = [tree_distance(i, anchor, depth, n_items) for i in probes]
    # closest (smallest tree_distance) first; stable ascending item order within ties
    order = sorted(range(len(probes)), key=lambda k: (dists[k], probes[k]))
    return [probes[k] for k in order], [dists[k] for k in order]


def _mode_specs(results):
    """(mode_idx, color, name) for the 4 modes shown throughout this figure --
    baseline, level 0 (Plant-Animal), and level 1's two modes shown
    SEPARATELY (Flower-Tree, which involves anchor=Daisy; Bird-Fish, which
    doesn't) rather than combined into one per-level panel."""
    levels = results["levels"]
    baseline_mode = int(np.where(levels == -1)[0][0])
    plant_animal_mode = int(np.where(levels == 0)[0][0])
    flower_tree_mode = A.anchor_relevant_mode(results, 1, ANCHOR)
    level1_idxs = np.where(levels == 1)[0]
    bird_fish_mode = int([m for m in level1_idxs if m != flower_tree_mode][0])
    return [
        (baseline_mode, BASELINE_COLOR, "Baseline"),
        (plant_animal_mode, PLANT_ANIMAL_COLOR, "Plant-Animal"),
        (flower_tree_mode, FLOWER_TREE_COLOR, "Flower-Tree"),
        (bird_fish_mode, BIRD_FISH_COLOR, "Bird-Fish"),
    ]


def _style_heatmap(fig, ax_heat, results, C, title, raw: bool, show_ylabel: bool = True):
    probes, _ = _probe_order(results, ANCHOR)
    C = C[np.ix_(probes, probes)]
    if not raw:
        C = C - C.mean()
    vmax = np.abs(C).max()
    vmax = vmax if vmax > 0 else 1.0
    im = ax_heat.imshow(C, cmap="PiYG", vmin=-vmax, vmax=vmax)
    names = [ITEM_NAMES[p] for p in probes]
    ax_heat.set_xticks(range(len(names)))
    ax_heat.set_xticklabels(names, fontsize=TICK_FS - 2, rotation=90)
    ax_heat.set_yticks(range(len(names)))
    ax_heat.set_yticklabels(names, fontsize=TICK_FS - 2)
    ax_heat.set_xlabel("Probe item $j$", fontsize=TICK_FS - 1)
    if show_ylabel:
        ax_heat.set_ylabel("Probe item $i$", fontsize=TICK_FS - 1)
    ax_heat.set_title(title, fontsize=TICK_FS - 1)
    cbar = fig.colorbar(im, ax=ax_heat, fraction=0.046, pad=0.04)
    cbar.set_label(r"$C^{(0)}_2(i,j)$", fontsize=TICK_FS - 2)
    cbar.ax.yaxis.offsetText.set_fontsize(TICK_FS - 2)
    cbar.ax.tick_params(labelsize=TICK_FS - 2)


def _plot_combined_modes(ax, results):
    """Row 2: all 4 modes' amplitude a_alpha(t) and co-update profile
    a_alpha(s_alpha-a_alpha) overlaid on one twin-axis plot."""
    ax_a = ax
    ax_b = ax.twinx()
    s = results["singular_values"]
    a_pre = results["a_pre"]

    for mode, color, name in _mode_specs(results):
        a = a_pre[:, mode]
        bump = a * (s[mode] - a)
        ax_a.plot(a, color=color, lw=1.6, ls="-", label=name)
        ax_b.plot(bump, color=color, lw=1.6, ls="--")

    ax_a.set_xlim(0, 2000)
    ax_a.set_xlabel("Trial", fontsize=LABEL_FS)
    ax_a.set_ylabel(r"$a_\alpha(t)$", fontsize=LABEL_FS)
    ax_b.set_ylabel(r"$a_\alpha(s_\alpha-a_\alpha)$", fontsize=LABEL_FS)
    ax_a.set_title("Per-mode amplitude and anchor-centered co-update profile", fontsize=TITLE_FS)
    ax_a.tick_params(labelsize=TICK_FS)
    ax_b.tick_params(axis="y", labelsize=TICK_FS)

    mode_legend = ax_a.legend(loc="upper left", frameon=False, fontsize=TICK_FS - 1, title="mode")
    ax_a.add_artist(mode_legend)
    style_handles = [
        Line2D([0], [0], color="black", lw=1.6, ls="-", label=r"$a_\alpha(t)$ (mode amplitude)"),
        Line2D([0], [0], color="black", lw=1.6, ls="--",
               label=r"$a_\alpha(s_\alpha{-}a_\alpha)$ (anchor centered co-update profile)"),
    ]
    ax_a.legend(handles=style_handles, loc="lower right", frameon=False, fontsize=TICK_FS - 1)


def _plot_mode_heatmaps(fig, gs_row, results):
    """Row 3: 4 heatmap panels, one per mode, each averaged over that mode's
    FULL active window (from when its bump first picks up until it falls
    back to baseline -- analysis.mode_bump_window), not a single peak trial."""
    for col, (mode, color, name) in enumerate(_mode_specs(results)):
        ax = fig.add_subplot(gs_row[0, col])
        lo, hi = A.mode_bump_window(results, mode)
        C = A.C_matrix_simulated_ranged(results, ANCHOR, lo, hi)
        raw = (name == "Baseline")
        _style_heatmap(fig, ax, results, C, f"{name}\ntrials {lo}-{hi}" + ("  (raw)" if raw else ""),
                       raw=raw, show_ylabel=(col == 0))


def _plot_probe_trajectories(ax, results):
    probes, dists = _probe_order(results, ANCHOR)
    labeled = set()
    for probe, d in zip(probes, dists):
        trials, mags = A.probe_coupdate_trajectory(results, ANCHOR, probe)
        color = DIST_COLORS[d]
        label = f"d={d}" if d not in labeled else None
        labeled.add(d)
        ax.plot(trials, mags, color=color, lw=1.0, alpha=0.85, label=label)
    ax.set_xlim(0, 2000)
    ax.set_xlabel("Trial", fontsize=LABEL_FS)
    ax.set_ylabel(r"$\|\Delta_0 h_2(i)\|$", fontsize=LABEL_FS)
    ax.set_title("Co-update trajectory per probe (color = tree distance to anchor)", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.legend(frameon=False, fontsize=TICK_FS - 1, title="distance", title_fontsize=TICK_FS - 1)


def make_figure(path: str = A.DEFAULT_PATH, out_path: str = DEFAULT_OUT_PATH):
    results = A.load_results(path)

    fig = plt.figure(figsize=(10.5, 12.5))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1.1, 1.3, 1], hspace=0.6, top=0.97, bottom=0.05,
                           left=0.07, right=0.93)  # 0.93 not 0.97: row 2's twinx right-axis
    # label needs the extra margin, otherwise it gets clipped past the figure edge

    gs_row1 = gs[0, 0].subgridspec(1, 3, wspace=0.7)
    SVD.plot_panels(fig, [fig.add_subplot(gs_row1[0, c]) for c in range(3)], results)

    _plot_combined_modes(fig.add_subplot(gs[1, 0]), results)

    gs_row3 = gs[2, 0].subgridspec(1, 4, wspace=0.6)
    _plot_mode_heatmaps(fig, gs_row3, results)

    _plot_probe_trajectories(fig.add_subplot(gs[3, 0]), results)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    make_figure()
