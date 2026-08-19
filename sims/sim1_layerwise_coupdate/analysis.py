"""Offline co-update analyses for Simulation 1.

For every trial t with anchor a_t, the shift vector of item i at layer l is
Delta_l(i, t) = post - pre. These functions quantify:
  1. the (expected-exact) zero shift of probes at h1 (orthogonal inputs),
  2. probe-shift magnitude over learning, per layer,
  3. anchor-probe shift correlation across trials, per layer,
  4. an NxN anchor-by-probe co-update matrix, per layer,
  5. the eq. 16 closed-form prediction for Delta_h2(i), validated against the
     simulated shift (orthogonal-input data only -- see predicted_h2_shift).
"""
import os
import sys

# Make `sims` importable regardless of the working directory the script is
# launched from (e.g. Spyder's %runfile --wdir cd's into this script's own
# folder, where `sims` isn't on sys.path).
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
from scipy import stats

DEFAULT_PATH_ORTHOGONAL = os.path.join(
    _PROJECT_ROOT, "outputs", "sim1_layerwise_coupdate", "results_orthogonal.npz"
)
DEFAULT_PATH_CORRELATED = os.path.join(
    _PROJECT_ROOT, "outputs", "sim1_layerwise_coupdate", "results_correlated.npz"
)
DEFAULT_PATH = DEFAULT_PATH_ORTHOGONAL  # backward-compatible alias

# results_*.npz array reference (T = n_trials, N = n_items):
#   h1_pre, h1_post   (T, N, h1_dim) float64  -- layer-1 hidden rep of every item, before/after trial t's update
#   h2_pre, h2_post   (T, N, h2_dim) float64  -- layer-2 hidden rep of every item, before/after
#   y_pre,  y_post    (T, N)         float64  -- scalar network output for every item, before/after
#   anchor_idx        (T,)           int64    -- which item (0..N-1) was trained/fed back on trial t
#   anchor_loss       (T,)           float64  -- loss on the anchor at trial t (pre-update forward pass)
#   W3_pre            (T, h2_dim)    float64  -- W3 weight vector, BEFORE trial t's update (for eq. 16)
#   X                 (N, input_dim) float64  -- fixed input vectors (constant across trials)
#   v                 (N,)           float64  -- fixed target scalar value per item ("value function")
#   W1_final/W2_final/W3_final       float64  -- trained weight matrices at end of run
#   n_items, input_dim, h1_dim, h2_dim, lr, n_trials, seed, rho  -- scalar Sim1Config fields (rho=nan if orthogonal)


def load_results(path: str = DEFAULT_PATH):
    return np.load(path, allow_pickle=True)


def shift_vectors(results, layer: str) -> np.ndarray:
    pre = results[f"{layer}_pre"]
    post = results[f"{layer}_post"]
    return post - pre  # (T, N, dim)


def shift_magnitudes(results, layer: str) -> np.ndarray:
    return np.linalg.norm(shift_vectors(results, layer), axis=-1)  # (T, N)


def probe_mask(results) -> np.ndarray:
    anchor_idx = results["anchor_idx"]
    n_items = results["h1_pre"].shape[1]
    T = anchor_idx.shape[0]
    mask = np.ones((T, n_items), dtype=bool)
    mask[np.arange(T), anchor_idx] = False
    return mask


def zero_shift_check(results, layer: str = "h1") -> float:
    """Max shift magnitude among probes (non-anchor items) at this layer."""
    mag = shift_magnitudes(results, layer)
    mask = probe_mask(results)
    return float(mag[mask].max())


def probe_shift_curve(results, layer: str) -> np.ndarray:
    """Per-trial mean shift magnitude across probes (excludes the anchor)."""
    mag = shift_magnitudes(results, layer)
    mask = probe_mask(results)
    masked = np.where(mask, mag, np.nan)
    return np.nanmean(masked, axis=1)  # (T,)


def anchor_shift_curve(results, layer: str) -> np.ndarray:
    mag = shift_magnitudes(results, layer)
    anchor_idx = results["anchor_idx"]
    return mag[np.arange(mag.shape[0]), anchor_idx]


def anchor_probe_correlation(results, layer: str):
    """Pearson r, p between per-trial anchor shift and mean probe shift.

    Returns (nan, nan) if either series has ~zero variance (as expected at
    h1, where probe shift is exactly zero by construction).
    """
    anchor_mag = anchor_shift_curve(results, layer)
    probe_mag = probe_shift_curve(results, layer)
    valid = np.isfinite(anchor_mag) & np.isfinite(probe_mag)
    a, p_series = anchor_mag[valid], probe_mag[valid]
    if np.std(a) < 1e-12 or np.std(p_series) < 1e-12:
        return float("nan"), float("nan")
    r, p = stats.pearsonr(a, p_series)
    return float(r), float(p)


def coupdate_matrix(results, layer: str) -> np.ndarray:
    """(n_items, n_items) matrix: mat[a, i] = mean shift magnitude of probe i
    across all trials where item a was the anchor. Diagonal (a == i) is NaN.
    """
    mag = shift_magnitudes(results, layer)  # (T, N)
    anchor_idx = results["anchor_idx"]
    n_items = mag.shape[1]
    mat = np.full((n_items, n_items), np.nan)
    for a in range(n_items):
        trials = anchor_idx == a
        if not trials.any():
            continue
        mat[a] = mag[trials].mean(axis=0)
        mat[a, a] = np.nan
    return mat


def predicted_h2_shift(results) -> np.ndarray:
    """Eq. 16 closed-form prediction for Delta_h2(i) on every trial:

        Delta_h2(i) = eta * r_a * K1(i, a) * W3_pre^T

    where r_a = v_a - y_a(pre) (target minus pre-update prediction),
    K1(i, a) = h1_i(pre) . h1_a(pre), and W3_pre is this trial's W3 BEFORE
    the update. Exact (not just first-order) under orthogonal inputs, since
    probes' h1 is provably unchanged there. Returns (T, N, h2_dim).
    """
    anchor_idx = results["anchor_idx"]
    T = anchor_idx.shape[0]
    eta = float(results["lr"])
    v = results["v"]
    y_pre = results["y_pre"]
    h1_pre = results["h1_pre"]
    W3_pre = results["W3_pre"]

    r_a = v[anchor_idx] - y_pre[np.arange(T), anchor_idx]  # (T,)
    h1_a = h1_pre[np.arange(T), anchor_idx]  # (T, h1_dim)
    K1 = np.einsum("tnd,td->tn", h1_pre, h1_a)  # (T, N)
    pred = (eta * r_a)[:, None, None] * K1[:, :, None] * W3_pre[:, None, :]  # (T, N, h2_dim)
    return pred


def eq16_validation_pool(results):
    """Pooled (predicted, actual) Delta_h2 entries across all trials, probes
    (anchor excluded), and h2 dimensions, plus Pearson r and R^2 over the
    full pool. Meant for the orthogonal-input results file.
    """
    actual = shift_vectors(results, "h2")  # (T, N, h2_dim)
    pred = predicted_h2_shift(results)  # (T, N, h2_dim)
    mask = probe_mask(results)  # (T, N)
    actual_flat = actual[mask].ravel()
    pred_flat = pred[mask].ravel()
    r, _ = stats.pearsonr(pred_flat, actual_flat)
    ss_res = np.sum((actual_flat - pred_flat) ** 2)
    ss_tot = np.sum((actual_flat - actual_flat.mean()) ** 2)
    r2 = 1.0 - ss_res / ss_tot
    return pred_flat, actual_flat, float(r), float(r2)


def pick_snapshot_trial(results, anchor: int, trial_lo: int, trial_hi: int) -> int:
    """First trial index in [trial_lo, trial_hi) where `anchor` was the
    trained item. Used by figure 2 to fix a single (anchor, trial) snapshot
    for the C^(a)_2 similarity-matrix analysis.
    """
    anchor_idx = results["anchor_idx"]
    lo, hi = trial_lo, min(trial_hi, anchor_idx.shape[0])
    window = np.arange(lo, hi)
    candidates = window[anchor_idx[window] == anchor]
    if candidates.size == 0:
        raise ValueError(f"anchor {anchor} not trained in trial window [{trial_lo}, {trial_hi})")
    return int(candidates[0])


def k_vector(results, trial_idx: int, anchor: int) -> np.ndarray:
    """K1(i, a) for every item i at this trial: h1_i(pre) . h1_a(pre). (N,)"""
    h1_pre = results["h1_pre"][trial_idx]  # (N, h1_dim)
    return h1_pre @ h1_pre[anchor]


def simulated_gram_matrix(results, trial_idx: int, layer: str = "h2") -> np.ndarray:
    """C^(a) from simulation: pairwise inner products of every item's shift
    vector at this trial. (N, N). Caller restricts to probes via exclude_anchor.
    """
    shifts = shift_vectors(results, layer)[trial_idx]  # (N, dim)
    return shifts @ shifts.T


def gram_scale_factor(results, trial_idx: int, anchor: int) -> float:
    """The positive scalar prefactor c in C^(a)_2 = c * k_a k_a^T, i.e.
    c = (eta * r_a * ||W3_pre||)^2. Computed entirely from eta, r_a, and
    W3_pre -- independent of k_a itself, so it can be used to derive k_a's
    magnitude from the eigenvalue without assuming what we're trying to show.
    """
    eta = float(results["lr"])
    r_a = float(results["v"][anchor] - results["y_pre"][trial_idx, anchor])
    W3_pre = results["W3_pre"][trial_idx]
    return (eta * r_a * np.linalg.norm(W3_pre)) ** 2


def predicted_gram_matrix(results, trial_idx: int, anchor: int) -> np.ndarray:
    """Eq. 16 closed form: C^(a)_2 = eta^2 * ||W3_pre . r_a||^2 * outer(k_a, k_a).
    (N, N). Caller restricts to probes via exclude_anchor.
    """
    k = k_vector(results, trial_idx, anchor)
    scale = gram_scale_factor(results, trial_idx, anchor)
    return scale * np.outer(k, k)


def exclude_anchor(arr: np.ndarray, anchor: int) -> np.ndarray:
    """Drop index `anchor` from the last axis/axes of a 1-D or 2-D array
    (vector -> (N-1,); square matrix -> (N-1,N-1)), restricting to probes.
    """
    keep = np.array([i for i in range(arr.shape[0]) if i != anchor])
    if arr.ndim == 1:
        return arr[keep]
    return arr[np.ix_(keep, keep)]


def leading_eigenpair(gram_matrix: np.ndarray, align_with: np.ndarray = None):
    """Largest-magnitude eigenvalue/eigenvector of a symmetric matrix.

    Eigenvectors are only defined up to sign; if `align_with` is given, the
    eigenvector's sign is flipped so its dot product with `align_with` is
    positive (needed to compare against k_a in figure 2 panel c).
    Returns (eigenvalues_desc, leading_eigenvalue, leading_eigenvector).
    """
    eigvals, eigvecs = np.linalg.eigh(gram_matrix)  # ascending order
    order = np.argsort(eigvals)[::-1]
    eigvals_desc = eigvals[order]
    leading_vec = eigvecs[:, order[0]]
    if align_with is not None and np.dot(leading_vec, align_with) < 0:
        leading_vec = -leading_vec
    return eigvals_desc, eigvals_desc[0], leading_vec


def run_report(path: str = DEFAULT_PATH):
    results = load_results(path)
    print("=== Simulation 1 co-update analysis report ===")
    for layer in ("h1", "h2"):
        max_probe_shift = zero_shift_check(results, layer)
        r, p = anchor_probe_correlation(results, layer)
        corr_str = f"r={r:.3f}, p={p:.3e}" if np.isfinite(r) else "undefined (zero probe-shift variance)"
        print(f"[{layer}] max probe-shift magnitude (expect ~0 at h1): {max_probe_shift:.3e}")
        print(f"[{layer}] anchor-probe shift correlation: {corr_str}")

    pred_flat, actual_flat, r, r2 = eq16_validation_pool(results)
    max_abs_err = float(np.abs(actual_flat - pred_flat).max())
    print(f"[eq16] Delta_h2 prediction vs. simulation: r={r:.6f}, R^2={r2:.6f}, "
          f"max abs error={max_abs_err:.3e}, n_points={pred_flat.size}")
    return results


def run_report_figure2(anchor: int = 0, trial_lo: int = 116, trial_hi: int = 124, path: str = DEFAULT_PATH):
    """Verification checks for figure 2: simulated vs. predicted C^(a)_2
    match, and the leading-to-second eigenvalue ratio (rank-1 confirmation).
    """
    results = load_results(path)
    trial_idx = pick_snapshot_trial(results, anchor, trial_lo, trial_hi)

    C_sim = exclude_anchor(simulated_gram_matrix(results, trial_idx), anchor)
    C_pred = exclude_anchor(predicted_gram_matrix(results, trial_idx, anchor), anchor)
    max_abs_err = float(np.abs(C_sim - C_pred).max())

    eigvals, lead_val, _ = leading_eigenpair(C_sim)
    second_val = eigvals[1]
    ratio = lead_val / second_val if second_val != 0 else float("inf")

    print("=== Figure 2 report ===")
    print(f"anchor={anchor}, trial_idx={trial_idx}")
    print(f"[C^(a)_2] simulated vs. predicted max abs error: {max_abs_err:.3e}")
    print(f"[eigenspectrum] leading={lead_val:.3e}, second={second_val:.3e}, ratio={ratio:.3e}")
    return results, trial_idx


if __name__ == "__main__":
    run_report()
    run_report_figure2()
