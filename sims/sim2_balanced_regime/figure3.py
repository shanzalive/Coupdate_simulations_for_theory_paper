"""Camera-ready paper figure: transient co-update waves in the balanced
regime -- learning decomposes into 3 sequentially-learned modes.

1 row x 3 columns:
  (a) learned mode amplitudes a_alpha(t) -- staggered sigmoids
  (b) per-mode layer-2 co-update amplitude a_alpha(t)(s_alpha - a_alpha(t)) -- sequence of bumps
  (c) K2(i,j;t) (saturating) vs C^(0)_2(i,j;t) (transient) for one probe pair,
      analytical curves with real simulated data overlaid (including the
      K2 drift from single-anchor SGD vs. the theory's full-batch-gradient-
      flow assumption -- shown as-is, not hidden).

Panel letters intentionally omitted -- added later in Illustrator.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from sims.sim1_layerwise_coupdate.figure1 import LABEL_FS, TICK_FS, TITLE_FS
from sims.sim2_balanced_regime import analysis as A

DEFAULT_OUT_PATH = os.path.join(
    _PROJECT_ROOT, "outputs", "sim2_balanced_regime", "figures", "figure3_balanced_waves.pdf"
)

# Sequential (dark->light = large s -> small s), not qualitative, so the
# color itself hints at mode magnitude ordering.
MODE_COLORS = ["#08519c", "#4292c6", "#9ecae1"]
# Warm-toned, deliberately far from MODE_COLORS' blue family (C_COLOR was
# previously a light blue that read as "another mode" next to panels a/b).
K2_COLOR = "#D55E00"  # vermillion
C_COLOR = "#CC79A7"   # pink/magenta


def _plot_mode_amplitudes(ax, results):
    """No legend drawn here -- make_figure draws ONE shared legend between
    this panel and _plot_mode_bumps, using this panel's (fuller) labels."""
    a = results["a_pre"]  # (T, n_modes)
    s = results["singular_values"]
    n_modes = a.shape[1]
    for alpha in range(n_modes):
        ax.plot(a[:, alpha], color=MODE_COLORS[alpha % len(MODE_COLORS)], lw=1.3,
                 label=rf"$\alpha$={alpha + 1} ($s$={s[alpha]:g})")
    ax.set_xlabel("Trial", fontsize=LABEL_FS)
    ax.set_ylabel(r"Mode amplitude $a_\alpha(t)$", fontsize=LABEL_FS)
    ax.set_title("Learned mode\namplitudes", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    return ax.get_legend_handles_labels()


def _plot_mode_bumps(ax, results):
    bumps = A.mode_bump_analytical(results)  # (T, n_modes)
    n_modes = bumps.shape[1]
    for alpha in range(n_modes):
        ax.plot(bumps[:, alpha], color=MODE_COLORS[alpha % len(MODE_COLORS)], lw=1.3)
    ax.set_xlabel("Trial", fontsize=LABEL_FS)
    ax.set_ylabel(r"$a_\alpha(t)\,(s_\alpha - a_\alpha(t))$", fontsize=LABEL_FS)
    ax.set_title("Per-mode co-update\namplitude", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)


def _plot_k2_vs_c(ax, results):
    K2_pred = A.K_ell_analytical(results, 2, A.PROBE_I, A.PROBE_J)
    K2_sim = A.K2_simulated(results, A.PROBE_I, A.PROBE_J)
    C_pred = A.C_analytical(results, A.ANCHOR, A.PROBE_I, A.PROBE_J)
    trials_sparse, C_sim = A.C_simulated_sparse(results, A.ANCHOR, A.PROBE_I, A.PROBE_J)

    ax.plot(K2_pred, color=K2_COLOR, lw=1.3)
    ax.plot(K2_sim, color=K2_COLOR, lw=0.8, ls=":")
    ax.set_xlabel("Trial", fontsize=LABEL_FS)
    ax.set_ylabel(r"$K_2(i,j;t)$", fontsize=LABEL_FS, color=K2_COLOR)
    ax.tick_params(axis="y", labelcolor=K2_COLOR, labelsize=TICK_FS)
    ax.tick_params(axis="x", labelsize=TICK_FS)
    ax.set_title(f"Similarity vs. co-update\n(a={A.ANCHOR}, i={A.PROBE_I}, j={A.PROBE_J})", fontsize=TITLE_FS)

    ax2 = ax.twinx()
    ax2.plot(C_pred, color=C_COLOR, lw=1.3)
    ax2.scatter(trials_sparse, C_sim, color=C_COLOR, s=6, alpha=0.5)
    ax2.set_ylabel(r"$C^{(0)}_2(i,j;t)$", fontsize=LABEL_FS, color=C_COLOR)
    ax2.tick_params(axis="y", labelcolor=C_COLOR, labelsize=TICK_FS)

    # "Factorized" legend: each variable (K2, C2) named once, with its two
    # lines (exact/simulated) grouped underneath, instead of repeating the
    # variable name in every entry. The header rows use an invisible handle
    # (color="none") so only the label text shows, no marker/line stub.
    legend_elements = [
        Line2D([0], [0], color="none", label=r"$K_2$:"),
        Line2D([0], [0], color=K2_COLOR, lw=1.3, label="exact"),
        Line2D([0], [0], color=K2_COLOR, lw=0.8, ls=":", label="simulated"),
        Line2D([0], [0], color="none", label=r"$C^{(0)}_2$:"),
        Line2D([0], [0], color=C_COLOR, lw=1.3, label="exact"),
        Line2D([0], [0], color=C_COLOR, lw=0, marker="o", markersize=4, alpha=0.5, label="simulated"),
    ]
    # Shifted right (bbox_to_anchor x=0.4, not the default flush-left) so it
    # doesn't sit on top of the rising curves at the left of this now-wide panel.
    ax.legend(
        handles=legend_elements, fontsize=TICK_FS, loc="upper left",
        bbox_to_anchor=(0.4, 1.0), frameon=True, facecolor="white", edgecolor="none",
        framealpha=1.0, handlelength=1.3, labelspacing=0.3, borderpad=0.4,
    )


def make_figure(path: str = A.DEFAULT_PATH, out_path: str = DEFAULT_OUT_PATH):
    results = A.load_results(path)

    fig = plt.figure(figsize=(7.2, 2.8))
    gs = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1), wspace=0.55,
                          left=0.08, right=0.92, top=0.8, bottom=0.18)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[0, 2])

    handles, labels = _plot_mode_amplitudes(ax0, results)
    _plot_mode_bumps(ax1, results)
    _plot_k2_vs_c(ax2, results)

    # One shared legend (mode alpha + s) between panels (a) and (b), instead
    # of repeating it in both. The gap between the two axes is narrow
    # (~0.11 figure-units) and, at the default legend fontsize/spacing, the
    # legend box itself was measured to be ~0.11 wide -- wider than the gap
    # -- so it always overlapped one side or the other regardless of anchor
    # point. Shrinking font + handle/text spacing (measured: fits in ~0.08)
    # is what actually fixes it, not just repositioning.
    fig.canvas.draw()
    pos0, pos1 = ax0.get_position(), ax1.get_position()
    edge_x = pos0.x1 + 0.008
    top_y = pos0.y0 + 0.98 * (pos0.y1 - pos0.y0)  # pulled up, above the curves' plateau
    leg = fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(edge_x, top_y),
                      frameon=False, fontsize=6, handlelength=1.0, handletextpad=0.4,
                      labelspacing=0.3, borderaxespad=0)
    # Verify it actually cleared panel (b) -- fail loudly rather than silently
    # ship an overlap again if a future font/label change widens it back out.
    fig.canvas.draw()
    leg_x1 = fig.transFigure.inverted().transform(leg.get_window_extent())[1, 0]
    assert leg_x1 < pos1.x0, f"shared legend (right edge {leg_x1:.3f}) overlaps panel b (starts {pos1.x0:.3f})"

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    make_figure()
