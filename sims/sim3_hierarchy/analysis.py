"""Offline analyses for Simulation 3 (binary hierarchy, anchor-centered
ultrametric co-update). Mirrors sim 2's analysis.py structure, generalized
from D global modes to the full P-1 localized tree wavelets (see train.py's
module docstring). L=3 throughout.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np

from sims.sim3_hierarchy.train import tree_distance

DEFAULT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "sim3_hierarchy", "results.npz")
L = 3
ANCHOR = 0


def load_results(path: str = DEFAULT_PATH):
    return np.load(path, allow_pickle=True)


def shift_vectors_h2(results) -> np.ndarray:
    return results["h2_post"] - results["h2_pre"]  # (T, N, h2_dim)


def K1_analytical(results, anchor: int, i: int) -> np.ndarray:
    """Eq. 29 (K1(i,a;t) = sum_alpha a_alpha(t)^(2/L) * V[alpha,i] * V[alpha,a]),
    summed over ALL modes -- not just one "anchor-centered" mode per level.
    With the exact hand-built Haar wavelets, only one mode per level was
    nonzero at the anchor, so restricting the sum to those was equivalent;
    with the empirical (finite-sample, noisy) SVD modes used now, loadings
    are dense (every mode has some nonzero value at every item), so the
    general sum is required -- modes with ~0 loading at the anchor simply
    contribute ~0 automatically, no special-casing needed. (T,).
    """
    a = results["a_pre"]  # (T, n_modes)
    V = results["V"]  # (n_modes, n_items)
    exponent = 2.0 / L
    a_modes = np.abs(a) ** exponent  # (T, n_modes) -- see sim2 note on abs() for negative a
    psi_i = V[:, i]  # (n_modes,)
    psi_a = V[:, anchor]  # (n_modes,)
    return a_modes @ (psi_i * psi_a)


def b_norm_sq_analytical(results, anchor: int) -> np.ndarray:
    """||b_a(t)||^2 (eq. 27 generalized): sum over ALL modes, but only
    anchor-centered ones (nonzero V[:,anchor]) contribute -- the rest have
    psi(anchor)=0 and drop out automatically. (T,).
    """
    a = results["a_pre"]  # (T, n_modes)
    s = results["singular_values"]  # (n_modes,) -- per-mode, already level-expanded
    V = results["V"]  # (n_modes, n_items)
    exponent = (L - 2.0) / L
    terms = (np.abs(a) ** exponent) * (s[None, :] - a) * V[:, anchor][None, :]  # (T, n_modes)
    return np.sum(terms ** 2, axis=1)


def C_analytical(results, anchor: int, i: int, j: int) -> np.ndarray:
    """Eq. 30 Gram entry: C^(a)_2(i,j;t) = eta^2 ||b_a(t)||^2 K1(i,a;t) K1(j,a;t). (T,)."""
    eta = float(results["lr"])
    b2 = b_norm_sq_analytical(results, anchor)
    K1_i = K1_analytical(results, anchor, i)
    K1_j = K1_analytical(results, anchor, j)
    return (eta ** 2) * b2 * K1_i * K1_j


def C_matrix_simulated(results, trial_idx: int) -> np.ndarray:
    """Full (n_items, n_items) simulated Gram matrix of h2 shifts at one
    specific trial (same idea as figure 2's simulated_gram_matrix)."""
    shifts = shift_vectors_h2(results)[trial_idx]  # (n_items, h2_dim)
    return shifts @ shifts.T


def C_matrix_simulated_ranged(results, anchor: int, lo: int, hi: int) -> np.ndarray:
    """Mean simulated Gram matrix over ALL anchor-trials with lo <= trial <= hi
    (inclusive), not just the n nearest trials to one point -- for a level
    whose co-update bump is wide (e.g. level 1/2, desynchronized across
    modes), a single-point window can land on an atypical trial; averaging
    over the entire active window (see analysis.mode_bump_analytical) gives
    the "typical" pattern for that level's whole active period instead.
    """
    anchor_idx = results["anchor_idx"]
    trial_num = np.arange(len(anchor_idx))
    chosen = np.where((anchor_idx == anchor) & (trial_num >= lo) & (trial_num <= hi))[0]
    assert len(chosen) > 0, f"no anchor-{anchor} trials in range [{lo}, {hi}]"
    mats = np.stack([C_matrix_simulated(results, int(t)) for t in chosen])
    return mats.mean(axis=0)


def probe_coupdate_trajectory(results, anchor: int, probe: int) -> tuple:
    """Simulated ||Delta_anchor h2(probe)|| at every trial where `anchor` was
    actually trained (the only trials where this co-update is defined) --
    a full time series, not a single snapshot. Returns (trial_indices, magnitudes).
    """
    anchor_idx = results["anchor_idx"]
    trials = np.where(anchor_idx == anchor)[0]
    shifts = shift_vectors_h2(results)[trials, probe, :]  # (n_anchor_trials, h2_dim)
    return trials, np.linalg.norm(shifts, axis=-1)


def mode_bump_analytical(results) -> np.ndarray:
    """Per-mode layer-2 co-update amplitude a_alpha(t)*(s_alpha - a_alpha(t)), eq. 28. (T, n_modes)."""
    a = results["a_pre"]  # (T, n_modes)
    s = results["singular_values"]  # (n_modes,)
    return a * (s[None, :] - a)


def mode_bump_window(results, mode: int, threshold: float = 0.15) -> tuple:
    """Trial range from when mode `mode`'s bump (a_alpha*(s_alpha-a_alpha))
    first rises above threshold*peak until it falls back below it -- the
    FULL active duration ("picks up until it returns to baseline"), not
    just a narrow window around the peak rate of change (mode_window /
    _contiguous_change_window). The bump metric itself (not its derivative)
    is ~0 both before the mode starts escaping the small-init plateau and
    after it settles, so a simple global min/max of threshold-exceeding
    trials works well for a SINGLE mode's own bump (unlike a level-wide
    mean bump, which can have multiple disjoint excursions from
    desynchronized sibling modes -- see level_active_window's history).
    """
    bump = mode_bump_analytical(results)[:, mode]
    peak = bump.max()
    active = np.where(bump > threshold * peak)[0]
    return int(active.min()), int(active.max())


def _contiguous_change_window(trace: np.ndarray, threshold: float = 0.4, smooth: int = 25) -> tuple:
    """Trial range around `trace`'s peak RATE OF CHANGE (smoothed, edge-padded
    derivative) -- the contiguous region around the single global peak
    derivative (walking outward from the peak while the derivative stays
    above threshold*peak), not all trials anywhere exceeding the threshold
    (the latter lets isolated noise elsewhere in the trace spuriously
    extend the window). Edge-padding before smoothing matters -- plain
    np.convolve(..., mode='same') distorts near the array boundary and can
    spuriously produce the largest apparent derivative right at the end of
    a converged trace.
    """
    pad = smooth // 2
    trace_padded = np.pad(trace, pad, mode="edge")
    kernel = np.ones(smooth) / smooth
    smoothed = np.convolve(trace_padded, kernel, mode="valid")  # same length as `trace`, no edge distortion
    deriv = np.abs(np.diff(smoothed, prepend=smoothed[0]))
    peak_trial = int(np.argmax(deriv))
    peak = deriv[peak_trial]
    lo = peak_trial
    while lo > 0 and deriv[lo - 1] > threshold * peak:
        lo -= 1
    hi = peak_trial
    while hi < len(deriv) - 1 and deriv[hi + 1] > threshold * peak:
        hi += 1
    return lo, hi


def level_active_window(results, level: int, threshold: float = 0.4, smooth: int = 25) -> tuple:
    """Trial range around the level's peak RATE OF CHANGE of its raw MEAN
    amplitude a_alpha(t) (averaged across all modes at that level) -- i.e.
    where a_alpha(t) is rising fastest, not where the transient bump metric
    a_alpha(s_alpha-a_alpha) is largest (switched 2026-08-18: at larger
    n_features, the bump metric's (s_alpha-a_alpha) factor amplifies
    per-trial noise; a_alpha(t) itself is far better-behaved).

    NOTE: averaging across ALL of a level's modes mixes in modes irrelevant
    to any specific anchor (e.g. level 1 has one mode that splits anchor 0's
    own branch and one that splits an unrelated branch entirely) -- for
    anchor-centered analysis, prefer anchor_mode_window instead, which
    isolates just the anchor-relevant mode. This level-wide version remains
    useful for level-level summaries not tied to one specific anchor (e.g.
    leakage_report).
    """
    levels = results["levels"]
    idxs = np.where(levels == level)[0]
    a = results["a_pre"][:, idxs].mean(axis=1)
    return _contiguous_change_window(a, threshold, smooth)


def anchor_relevant_mode(results, level: int, anchor: int) -> int:
    """The single mode (index into the n_modes axis) at `level` most
    relevant to `anchor` -- the one with the largest |loading| at the
    anchor item. At level 0 there's only one mode total (the coarsest
    split spans every item), so this is trivial there; it matters starting
    at level 1, where e.g. one of 2 modes splits the anchor's own branch
    and the other splits a branch the anchor isn't even in.
    """
    V = results["V"]
    levels = results["levels"]
    idxs = np.where(levels == level)[0]
    return int(idxs[np.argmax(np.abs(V[idxs, anchor]))])


def mode_window(results, mode: int, threshold: float = 0.4, smooth: int = 25) -> tuple:
    """Trial range around the peak rate of change of ONE specific mode
    (by raw index into the n_modes axis), regardless of level or anchor
    relevance -- the building block anchor_mode_window and per-mode-peak
    averaging (see figure4.py) are both built from.
    """
    a = results["a_pre"][:, mode]
    return _contiguous_change_window(a, threshold, smooth)


def anchor_mode_window(results, level: int, anchor: int, threshold: float = 0.4, smooth: int = 25) -> tuple:
    """Trial range around the peak rate of change of JUST the mode most
    relevant to `anchor` at this level (level=-1 for baseline, which always
    has exactly one mode) -- see anchor_relevant_mode's docstring for why
    this, not the level-wide mean, is the right window for anchor-centered
    co-update analysis.
    """
    mode = anchor_relevant_mode(results, level, anchor)
    return mode_window(results, mode, threshold, smooth)


def leakage_report(results, anchor: int = ANCHOR) -> list:
    """For each level's active window, mean ||Delta_anchor h2(i)|| grouped
    by tree_distance -- quantifies co-update "leaking" into items outside
    that level's anchor-centered wavelet block (idealized theory predicts
    ~0 there; nonzero reflects shared hidden-layer capacity / random-init
    gauge effects, a real simulation phenomenon, not a bug -- see project
    memory). Returns a list of (level, lo, hi, {distance: mean_magnitude}).
    """
    depth = int(results["tree_depth"])
    n_items = int(results["n_items"])
    report = []
    for lam in range(depth):
        lo, hi = level_active_window(results, lam)
        by_dist = {}
        for i in range(n_items):
            if i == anchor:
                continue
            d = tree_distance(i, anchor, depth, n_items)
            trials, mags = probe_coupdate_trajectory(results, anchor, i)
            mask = (trials >= lo) & (trials <= hi)
            by_dist.setdefault(d, []).append(float(mags[mask].mean()))
        by_dist = {d: float(np.mean(v)) for d, v in sorted(by_dist.items())}
        report.append((lam, lo, hi, by_dist))
    return report


def run_report(path: str = DEFAULT_PATH):
    results = load_results(path)
    a_final = results["a_pre"][-1]
    levels = results["levels"]
    s = results["singular_values"]
    depth = int(results["tree_depth"])
    print("=== Simulation 3 (hierarchy) report ===")
    # Empirical (finite-sample) modes don't share an exact singular value
    # within a level anymore (unlike the old hand-built wavelets) -- report
    # each mode's own target vs. final amplitude, grouped by level for
    # readability.
    for lam in list(range(depth)) + [-1]:
        mask = levels == lam
        label = f"level {lam}" if lam >= 0 else "baseline"
        print(f"{label}: target s = {np.round(s[mask], 3)}, final a = {np.round(a_final[mask], 3)}")

    # Validate C^(0)_2 analytical vs. simulated for a same-pair probe (item 1)
    # vs. a different-branch probe (item 4), both relative to anchor 0.
    trials = np.where(results["anchor_idx"] == ANCHOR)[0]
    shifts = shift_vectors_h2(results)
    for probe in (1, 4):
        C_sim = np.einsum("td,td->t", shifts[trials, probe, :], shifts[trials, ANCHOR, :])
        C_pred = C_analytical(results, ANCHOR, probe, ANCHOR)[trials]
        max_err = float(np.abs(C_sim - C_pred).max())
        corr = float(np.corrcoef(C_sim, C_pred)[0, 1])
        dist = tree_distance(probe, ANCHOR, int(results["tree_depth"]), int(results["n_items"]))
        print(f"[C^(0)_2({probe},0)] tree_distance={dist}: analytical vs. simulated "
              f"(n={len(trials)}): max abs err={max_err:.3e}, corr r={corr:.6f}")

    # Leakage check: idealized theory predicts ~0 co-update for items outside
    # a level's anchor-centered wavelet block; report what's actually there.
    print("\n--- Leakage check (mean ||Delta_0 h2(i)|| by tree_distance, per level's active window) ---")
    for lam, lo, hi, by_dist in leakage_report(results):
        farthest, closest = max(by_dist), min(by_dist)
        ratio = by_dist[farthest] / by_dist[closest] if by_dist[closest] > 0 else float("nan")
        print(f"level {lam} (trials {lo}-{hi}): {by_dist}  "
              f"[farthest(d={farthest})/closest(d={closest}) ratio = {ratio:.3f}]")
    return results


if __name__ == "__main__":
    run_report()
