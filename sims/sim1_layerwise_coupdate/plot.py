"""Figures for Simulation 1 (co-update in a depth-3 deep linear network)."""
import os
import sys

# Make `sims` importable regardless of the working directory the script is
# launched from (e.g. Spyder's %runfile --wdir cd's into this script's own
# folder, where `sims` isn't on sys.path).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import matplotlib.pyplot as plt
import numpy as np

from sims.sim1_layerwise_coupdate import analysis as A

DEFAULT_FIG_DIR = os.path.join(_PROJECT_ROOT, "outputs", "sim1_layerwise_coupdate", "figures")
LAYER_COLORS = {"h1": "#4C72B0", "h2": "#C44E52"}


def plot_learning_curve(results, out_dir: str):
    loss = results["anchor_loss"]

    fig, ax = plt.subplots(figsize=(4, 3))
    ax.plot(loss, color="k")
    ax.set_xlabel("trial")
    ax.set_ylabel("anchor loss")
    ax.set_title("Learning curve")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "learning_curve.pdf"))
    plt.close(fig)


def plot_probe_shift_timeseries(results, out_dir: str):
    fig, ax = plt.subplots(figsize=(4.5, 3))
    for layer in ("h1", "h2"):
        curve = A.probe_shift_curve(results, layer)
        ax.plot(curve, label=layer, color=LAYER_COLORS[layer])
    ax.set_xlabel("trial")
    ax.set_ylabel("mean probe shift magnitude")
    ax.set_title("Probe representational shift, per layer")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "probe_shift_timeseries.pdf"))
    plt.close(fig)


def plot_anchor_probe_scatter(results, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3))
    for ax, layer in zip(axes, ("h1", "h2")):
        anchor_mag = A.anchor_shift_curve(results, layer)
        probe_mag = A.probe_shift_curve(results, layer)
        r, p = A.anchor_probe_correlation(results, layer)
        ax.scatter(anchor_mag, probe_mag, s=6, alpha=0.4, color=LAYER_COLORS[layer])
        title = f"{layer}: r={r:.2f}, p={p:.1e}" if np.isfinite(r) else f"{layer}: undefined (zero variance)"
        ax.set_title(title, fontsize=9)
        ax.set_xlabel("anchor shift magnitude")
        ax.set_ylabel("mean probe shift magnitude")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "anchor_probe_scatter.pdf"))
    plt.close(fig)


def plot_coupdate_matrices(results, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, layer in zip(axes, ("h1", "h2")):
        mat = A.coupdate_matrix(results, layer)
        im = ax.imshow(mat, cmap="viridis")
        ax.set_title(f"{layer} co-update matrix")
        ax.set_xlabel("probe item")
        ax.set_ylabel("anchor item")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "coupdate_matrices.pdf"))
    plt.close(fig)


def make_all_figures(path: str = A.DEFAULT_PATH, out_dir: str = DEFAULT_FIG_DIR):
    results = A.load_results(path)
    os.makedirs(out_dir, exist_ok=True)
    plot_learning_curve(results, out_dir)
    plot_probe_shift_timeseries(results, out_dir)
    plot_anchor_probe_scatter(results, out_dir)
    plot_coupdate_matrices(results, out_dir)
    print(f"Figures saved to {out_dir}")


if __name__ == "__main__":
    make_all_figures()
