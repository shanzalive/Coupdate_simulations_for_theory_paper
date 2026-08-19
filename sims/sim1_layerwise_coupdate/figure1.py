"""Camera-ready paper figure: layer-wise co-update, orthogonal vs. correlated
inputs, plus eq. 16 validation.

2 rows x 3 columns:
  Row 1 (orthogonal inputs):  input vectors, co-update timeseries, eq. 16 validation
  Row 2 (correlated inputs):  input vectors, co-update timeseries, (col 3 unused)

Panel letters are intentionally omitted -- added later in Illustrator.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from sims.sim1_layerwise_coupdate import analysis as A

DEFAULT_OUT_PATH = os.path.join(
    _PROJECT_ROOT, "outputs", "sim1_layerwise_coupdate", "figures", "figure1_coupdate.pdf"
)
LAYER_COLORS = {"h1": "#4C72B0", "h2": "#C44E52"}
LABEL_COLORMAP = ListedColormap(["black", "white"])  # [-1, +1]

TITLE_FS = 10
LABEL_FS = 8
TICK_FS = 7

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]


# Note: no block/epoch averaging -- trial is the only unit, so these are
# raw per-trial curves (noisier than a smoothed version, but unambiguous).


def _plot_input_heatmap(fig, gs_cell, results, title: str):
    X = results["X"]  # (n_items, input_dim)
    v = results["v"]  # (n_items,)

    # [colorbar | heatmap+label] ; heatmap and label are further split, kept
    # close together, with the label column twice the colorbar's width.
    outer = gs_cell.subgridspec(1, 2, width_ratios=[1, 10], wspace=0.35)
    ax_cbar = fig.add_subplot(outer[0, 0])
    inner = outer[0, 1].subgridspec(1, 2, width_ratios=[8, 2], wspace=0.55)
    ax_main = fig.add_subplot(inner[0, 0])
    ax_label = fig.add_subplot(inner[0, 1])

    n_items = X.shape[0]
    row_boundaries = np.arange(0.5, n_items - 0.5, 1)

    vmax = np.abs(X).max()
    im = ax_main.imshow(X, aspect="auto", cmap="PiYG", vmin=-vmax, vmax=vmax)
    for y in row_boundaries:
        ax_main.axhline(y, color="black", linewidth=0.6)
    ax_main.set_yticks([])
    ax_main.set_xlabel("Input dim", fontsize=LABEL_FS)
    ax_main.set_ylabel("Input", fontsize=LABEL_FS)
    ax_main.set_title(title, fontsize=TITLE_FS)
    ax_main.tick_params(labelsize=TICK_FS)

    ax_label.imshow((v > 0).astype(int).reshape(-1, 1), aspect="auto", cmap=LABEL_COLORMAP, vmin=0, vmax=1)
    for y in row_boundaries:
        ax_label.axhline(y, color="black", linewidth=0.6)
    ax_label.set_xticks([])
    ax_label.set_yticks([])
    ax_label.set_ylabel("Output target", fontsize=LABEL_FS)

    neg_val = v[v < 0][0] if (v < 0).any() else -1
    pos_val = v[v > 0][0] if (v > 0).any() else 1
    legend_handles = [
        Patch(facecolor="black", label=f"{neg_val:g}"),
        Patch(facecolor="white", edgecolor="black", linewidth=0.8, label=f"{pos_val:g}"),
    ]
    ax_label.legend(
        handles=legend_handles, loc="center left", bbox_to_anchor=(1.6, 0.5),
        frameon=False, fontsize=TICK_FS, handlelength=0.9, handleheight=0.9,
    )

    cbar = fig.colorbar(im, cax=ax_cbar)
    cbar.ax.yaxis.set_ticks_position("left")
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.tick_params(labelsize=TICK_FS)


def _plot_timeseries_panel(ax, results, title: str):
    for layer in ("h1", "h2"):
        ax.plot(A.probe_shift_curve(results, layer), label=layer, color=LAYER_COLORS[layer], lw=1.3)
    ax.set_xlabel("Trial", fontsize=LABEL_FS)
    ax.set_ylabel("Mean probe co-update norm\n" + r"$\langle \|\Delta_a h_\ell(x_i)\| \rangle_i$", fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

    ax2 = ax.twinx()
    ax2.plot(results["anchor_loss"], label="loss", color="#555555", lw=1.0, ls=":")
    ax2.set_ylabel("Loss", fontsize=LABEL_FS)
    ax2.tick_params(labelsize=TICK_FS)

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(
        lines1 + lines2, labels1 + labels2, fontsize=TICK_FS, loc="upper right",
        frameon=True, facecolor="white", edgecolor="none", framealpha=1.0,
        handlelength=1.3, labelspacing=0.3, borderpad=0.4,
    )


def _plot_eq16_scatter(ax, results_ortho, title: str):
    pred_flat, actual_flat, r, r2 = A.eq16_validation_pool(results_ortho)
    ax.scatter(pred_flat, actual_flat, s=2, alpha=0.15, color="#4C72B0", rasterized=True)

    lo = min(pred_flat.min(), actual_flat.min())
    hi = max(pred_flat.max(), actual_flat.max())
    pad = 0.03 * (hi - lo)
    lims = (lo - pad, hi + pad)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1, label="identity")

    # Identical ticks on both axes so the identity line reads as a true diagonal.
    ticks = MaxNLocator(nbins=3).tick_values(lo, hi)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.ticklabel_format(axis="both", style="sci", scilimits=(0, 0), useMathText=True)
    # x-axis offset text auto-placement collides with the title/xlabel in this
    # compact, equal-aspect panel -- replaced with a manual bottom-right
    # annotation. The y-axis offset text (auto-placed above the axes, top
    # left) does not collide with anything, so it's kept as-is.
    ax.xaxis.offsetText.set_visible(False)
    ax.yaxis.offsetText.set_fontsize(TICK_FS)
    ax.text(0.97, 0.03, r"$\times 10^{-3}$", transform=ax.transAxes, fontsize=TICK_FS, ha="right", va="bottom")

    ax.set_xlabel(r"Predicted $\Delta h_2(X_i)$ (eq. 16)", fontsize=LABEL_FS)
    ax.set_ylabel(r"Simulated $\Delta h_2(X_i)$", fontsize=LABEL_FS)
    ax.set_title(title, fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.text(0.05, 0.95, f"Pearson's $r$ = {r:.0f}", transform=ax.transAxes, fontsize=LABEL_FS, va="top")


def make_figure(
    orthogonal_path: str = A.DEFAULT_PATH_ORTHOGONAL,
    correlated_path: str = A.DEFAULT_PATH_CORRELATED,
    out_path: str = DEFAULT_OUT_PATH,
    width_ratios=(1, 1, 1),
):
    results_ortho = A.load_results(orthogonal_path)
    results_corr = A.load_results(correlated_path)
    rho = float(results_corr["rho"])

    fig = plt.figure(figsize=(7.2, 5.12))
    gs = fig.add_gridspec(
        2, 3, width_ratios=width_ratios, wspace=0.95, hspace=0.35,
        left=0.08, right=0.97, top=0.92, bottom=0.1,
    )

    # Row 1: orthogonal inputs
    _plot_input_heatmap(fig, gs[0, 0], results_ortho, "Orthogonal inputs")
    _plot_timeseries_panel(fig.add_subplot(gs[0, 1]), results_ortho, "Co-update, orthogonal inputs")
    _plot_eq16_scatter(
        fig.add_subplot(gs[0, 2]), results_ortho, "Simulated vs. predicted\nprobe co-update"
    )

    # Row 2: correlated inputs (col 3 intentionally left empty)
    _plot_input_heatmap(fig, gs[1, 0], results_corr, f"Correlated inputs ($\\rho$={rho:g})")
    _plot_timeseries_panel(fig.add_subplot(gs[1, 1]), results_corr, "Co-update, correlated inputs")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    make_figure()
