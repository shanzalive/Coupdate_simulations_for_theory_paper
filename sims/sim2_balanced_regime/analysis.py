"""Offline analyses for Simulation 2 (balanced-regime co-update waves).

All closed-form quantities here follow from a_alpha(t) (recorded directly
during training) and the known mode structure (V, singular_values) -- see
sims/sim2_balanced_regime/train.py's module docstring for the construction.
L=3 (network depth) throughout, matching train.py.

Where possible, closed-form predictions are checked against the real
simulated network (the same "simulate, then validate against theory" style
as sim 1/2's eq. 16 / C^(a)_2 checks).
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

DEFAULT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "sim2_balanced_regime", "results.npz")
L = 3

# Probe pair used for panel (c) -- both != the fixed anchor.
ANCHOR = 0
PROBE_I, PROBE_J = 1, 2


def load_results(path: str = DEFAULT_PATH):
    return np.load(path, allow_pickle=True)


def shift_vectors_h2(results) -> np.ndarray:
    """Delta_h2(i,t) = h2_post - h2_pre for every item, every trial. (T, N, h2_dim)."""
    return results["h2_post"] - results["h2_pre"]


def mode_bump_analytical(results) -> np.ndarray:
    """Eq. 28: per-mode layer-2 co-update amplitude a_alpha(t)*(s_alpha - a_alpha(t)). (T, n_modes)."""
    a = results["a_pre"]  # (T, n_modes)
    s = results["singular_values"]  # (n_modes,)
    return a * (s[None, :] - a)


def K_ell_analytical(results, ell: int, i: int, j: int) -> np.ndarray:
    """Eq. 25: K_ell(i,j;t) = sum_alpha a_alpha(t)^(2*ell/L) * v_alpha[i] * v_alpha[j]. (T,).

    Uses |a|^exponent: with random (not deliberately positive) init, a_alpha
    can dip slightly negative before committing to positive growth. Since L=3
    and 2*ell is always even, the mathematically correct real value of
    a^(2*ell/L) for negative a is |a|^(2*ell/L) regardless of sign (it's an
    even power of the real cube root) -- not NaN, which is what Python's **
    would give for a negative base with a non-integer exponent.
    """
    a = results["a_pre"]  # (T, n_modes)
    V = results["V"]  # (n_modes, n_items)
    exponent = 2.0 * ell / L
    return np.sum((np.abs(a) ** exponent) * V[:, i] * V[:, j], axis=1)


def b_norm_sq_analytical(results, anchor: int) -> np.ndarray:
    """||b_a(t)||^2 from eq. 27 -- sum over modes since the rho_{alpha,2} are
    orthonormal, so cross-mode terms vanish. (T,).

    Uses |a|^exponent here too: this exponent (1/3) *is* sign-preserving in
    general, but the result is squared immediately below (`terms ** 2`), so
    the sign washes out either way -- using abs() avoids NaN with no change
    to the final value.
    """
    a = results["a_pre"]  # (T, n_modes)
    s = results["singular_values"]  # (n_modes,)
    V = results["V"]  # (n_modes, n_items)
    exponent = (L - 2.0) / L
    terms = (np.abs(a) ** exponent) * (s[None, :] - a) * V[:, anchor][None, :]  # (T, n_modes)
    return np.sum(terms ** 2, axis=1)


def C_analytical(results, anchor: int, i: int, j: int) -> np.ndarray:
    """Eq. 27 Gram entry: C^(a)_2(i,j;t) = eta^2 * ||b_a(t)||^2 * K1(i,a;t) * K1(j,a;t). (T,)."""
    eta = float(results["lr"])
    b2 = b_norm_sq_analytical(results, anchor)
    K1_ia = K_ell_analytical(results, 1, i, anchor)
    K1_ja = K_ell_analytical(results, 1, j, anchor)
    return (eta ** 2) * b2 * K1_ia * K1_ja


def K2_simulated(results, i: int, j: int) -> np.ndarray:
    """Actual (simulated) h2-space similarity between two items, every trial. (T,)."""
    h2_pre = results["h2_pre"]
    return np.einsum("td,td->t", h2_pre[:, i, :], h2_pre[:, j, :])


def C_simulated_sparse(results, anchor: int, i: int, j: int):
    """Actual (simulated) C^(a)_2(i,j) Gram entry, only at the trials where
    `anchor` was actually the trained item (sparse -- ~1/n_items of trials).
    Returns (trial_indices, values).
    """
    anchor_idx = results["anchor_idx"]
    trials = np.where(anchor_idx == anchor)[0]
    shifts = shift_vectors_h2(results)
    vals = np.einsum("td,td->t", shifts[trials, i, :], shifts[trials, j, :])
    return trials, vals


def run_report(path: str = DEFAULT_PATH):
    results = load_results(path)
    a_final = results["a_pre"][-1]
    s = results["singular_values"]
    print("=== Simulation 2 (balanced regime) report ===")
    print(f"Final mode amplitudes a(T) = {np.round(a_final, 4)}, target s = {s}")

    trials, C_sim = C_simulated_sparse(results, ANCHOR, PROBE_I, PROBE_J)
    C_pred_all = C_analytical(results, ANCHOR, PROBE_I, PROBE_J)
    C_pred = C_pred_all[trials]
    max_abs_err = float(np.abs(C_sim - C_pred).max())
    corr = float(np.corrcoef(C_sim, C_pred)[0, 1]) if len(trials) > 1 else float("nan")
    print(f"[C^({ANCHOR})_2({PROBE_I},{PROBE_J})] analytical vs. simulated (n={len(trials)} sparse trials): "
          f"max abs error={max_abs_err:.3e}, corr r={corr:.6f}")

    K2_pred = K_ell_analytical(results, 2, PROBE_I, PROBE_J)
    K2_sim = K2_simulated(results, PROBE_I, PROBE_J)
    k2_max_err = float(np.abs(K2_pred - K2_sim).max())
    print(f"[K2({PROBE_I},{PROBE_J})] analytical vs. simulated (all {len(K2_pred)} trials): "
          f"max abs error={k2_max_err:.3e}")
    return results


if __name__ == "__main__":
    run_report()
