""" paper figure 2: probe-pair co-update similarity structure is
rank-1, for a fixed anchor and a single fixed trial.

1 row x 3 columns:
  (a) heatmap of the simulated C^(a)_2 (probes only, anchor excluded)
  (b) eigenspectrum of C^(a)_2 -- one nonzero eigenvalue
  (c) leading eigenvector vs. k_a := K1(:, a); identity line

Panel letters are intentionally omitted -- added later in Illustrator.
Reuses the depth-3/orthogonal-input simulation from sim 1 directly
(results_orthogonal.npz) -- no new training run.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from sims.sim1_layerwise_coupdate import analysis as A
from sims.sim1_layerwise_coupdate.figure1 import LABEL_FS, TICK_FS, TITLE_FS

DEFAULT_OUT_PATH = os.path.join(
    _PROJECT_ROOT, "outputs", "sim1_layerwise_coupdate", "figures", "figure2_rank1.pdf"
)

ANCHOR = 0
TRIAL_LO, TRIAL_HI = 116, 124  # mid-training (n_trials=240)

# Both figure 1 and figure 2 display magnitudes on the same x10^-3 basis,
# even though C^(a)_2's raw values are ~1e-5/1e-6 (products of Delta_h2-scale
# quantities) -- values are pre-scaled by SCALE before plotting rather than
# relying on matplotlib's auto-picked exponent, which would otherwise differ
# per panel.
SCALE = 1e3
SCALE_LABEL = r"$\times 10^{-3}$"


def _plot_gram_heatmap(ax, C, probe_labels, anchor: int):
    C_scaled = C * SCALE
    vmax = np.abs(C_scaled).max()
    im = ax.imshow(C_scaled, cmap="PiYG", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(probe_labels)))
    ax.set_yticks(range(len(probe_labels)))
    ax.set_xticklabels(probe_labels, fontsize=TICK_FS)
    ax.set_yticklabels(probe_labels, fontsize=TICK_FS)
    ax.set_xlabel("Probe j", fontsize=LABEL_FS)
    ax.set_ylabel("Probe i", fontsize=LABEL_FS)
    ax.set_title(rf"Simulated $C^{{(a)}}_2$ (a = item {anchor})", fontsize=TITLE_FS)
    cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=TICK_FS)
    # On the colorbar's own side (standard placement), not above (would
    # collide with the now-longer main title) or below (collides with the
    # x-tick labels).
    cbar.set_label(SCALE_LABEL, fontsize=TICK_FS, rotation=270, labelpad=10)


def _plot_eigenspectrum(ax, eigvals):
    # Raw (not x1e3): the dominant eigenvalue is ~5e-5, so even x1e3 only
    # reaches ~0.05, not the ~O(1) range the shared x10^-3 convention (see
    # module docstring) was meant for -- that convention doesn't fit this
    # panel's natural scale. Let matplotlib pick its own scientific-notation
    # offset instead of a manual, mismatched one.
    idx = np.arange(1, len(eigvals) + 1)
    ax.bar(idx, eigvals, color="#888888")
    ax.set_xticks(idx)
    ax.set_box_aspect(1)
    ax.set_xlabel("Eigenvalue index", fontsize=LABEL_FS)
    ax.set_ylabel(r"Eigenvalue of $C^{(a)}_2$", fontsize=LABEL_FS)
    ax.set_title("Eigenspectrum", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
    # Default offset-text placement (top-left of axes) collides with the
    # centered title above it -- move it into the top-right corner instead.
    # Force a draw first: the offset text string isn't computed until draw
    # time, so querying it immediately after ticklabel_format would be empty.
    ax.figure.canvas.draw()
    offset_str = ax.yaxis.get_offset_text().get_text()
    ax.yaxis.offsetText.set_visible(False)
    ax.text(0.98, 0.95, offset_str, transform=ax.transAxes, fontsize=TICK_FS, ha="right", va="top")


def _plot_eigenvector_scatter(ax, eigvec_scaled, k_a):
    r, _ = stats.pearsonr(k_a, eigvec_scaled)

    ax.scatter(k_a, eigvec_scaled, s=30, color="#888888", edgecolor="black", linewidth=0.7)
    lo = min(k_a.min(), eigvec_scaled.min())
    hi = max(k_a.max(), eigvec_scaled.max())
    pad = 0.08 * (hi - lo)
    lims = (lo - pad, hi + pad)
    ax.plot([lo, hi], [lo, hi], "k--", lw=1)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("$K_a$ values\n(anchor-probe i similarity in h1)", fontsize=LABEL_FS)
    ax.set_ylabel("Eigenvector values", fontsize=LABEL_FS)
    ax.set_title(r"Eigenvector vs. $k_a$", fontsize=TITLE_FS)
    ax.tick_params(labelsize=TICK_FS)
    ax.text(0.05, 0.95, f"Pearson's $r$ = {r:.0f}", transform=ax.transAxes, fontsize=LABEL_FS, va="top")


def make_figure(
    path: str = A.DEFAULT_PATH_ORTHOGONAL,
    anchor: int = ANCHOR,
    trial_lo: int = TRIAL_LO,
    trial_hi: int = TRIAL_HI,
    out_path: str = DEFAULT_OUT_PATH,
):
    results = A.load_results(path)
    trial_idx = A.pick_snapshot_trial(results, anchor, trial_lo, trial_hi)
    n_items = results["h1_pre"].shape[1]

    C_sim = A.exclude_anchor(A.simulated_gram_matrix(results, trial_idx), anchor)
    k_a = A.exclude_anchor(A.k_vector(results, trial_idx, anchor), anchor)
    eigvals, _, lead_vec = A.leading_eigenpair(C_sim, align_with=k_a)

    # Rescale the (unit-norm, by eigh's convention) eigenvector using the
    # eigenvalue and the theoretical scale factor c = (eta*r_a*||W3_pre||)^2
    # -- both independent of k_a's own norm -- rather than just multiplying
    # by ||k_a|| directly, which would trivially force the scales to match.
    c = A.gram_scale_factor(results, trial_idx, anchor)
    eigvec_scaled = lead_vec * np.sqrt(eigvals[0] / c)

    probe_labels = [str(i) for i in range(n_items) if i != anchor]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    fig.subplots_adjust(left=0.08, right=0.97, top=0.82, bottom=0.18, wspace=0.65)

    _plot_gram_heatmap(axes[0], C_sim, probe_labels, anchor)
    _plot_eigenspectrum(axes[1], eigvals)
    _plot_eigenvector_scatter(axes[2], eigvec_scaled, k_a)

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    print(f"Figure saved to {out_path}")


if __name__ == "__main__":
    make_figure()
