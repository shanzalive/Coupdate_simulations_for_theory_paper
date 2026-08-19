"""Train a depth-3 deep linear network from a small random Gaussian
initialization -- matching Saxe et al. 2019 PNAS's actual simulation
methodology (SI: "networks were initialized with independent random
Gaussian weights", NOT deliberately aligned to the task's SVD directions)
-- using the exact same single-anchor-per-trial curriculum as sim 1/2
(shuffled queue, one item's feedback per trial). Unlike sim 1/2, the output
is multi-dimensional (one slot per hierarchy level / mode), so the task has
multiple singular values and the network learns them in a staggered
sequence. The "balanced regime" (eq. 23) is an approximation that emerges
from smallness + randomness, not something built into the initialization --
see analysis.py/figure3.py for how well the emergent trajectory matches the
closed-form aligned-regime predictions.

Task construction: n_items=8 items sit at the leaves of a depth-3 binary
hierarchy (8 = 2^3). The 3 "modes" are the 3 levels of that hierarchy --
mode 1 splits the items into two groups of 4 (coarsest, learned first),
mode 2 splits each group of 4 into pairs, mode 3 splits each pair into
individuals (finest, learned last). Each mode's item-loading pattern v_alpha
is a length-8 +-1 (Hadamard/Walsh) vector matching one bit of the item's
index; these are exactly orthonormal by construction. Output dimension
equals the number of modes, with u_alpha = the alpha-th standard basis
vector -- the simplest possible choice, so output slot alpha directly reads
out hierarchy level alpha.
"""
import os
import sys

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import numpy as np
import torch

from sims.common.inputs import make_orthonormal_inputs
from sims.common.linear_net import DeepLinearNet
from sims.sim2_balanced_regime.config import Sim2Config

DEFAULT_OUT_PATH = os.path.join(_PROJECT_ROOT, "outputs", "sim2_balanced_regime", "results.npz")

L = 3  # network depth (W1, W2, W3) -- matches DeepLinearNet, not a config knob


def hadamard_v_matrix(n_items: int, n_modes: int) -> np.ndarray:
    """(n_modes, n_items) matrix of orthonormal +-1 (Hadamard/Walsh) item-
    loading patterns, one row per mode, mode alpha's sign = bit (n_modes-1-alpha)
    of the item index. Requires n_items == 2**n_modes.
    """
    assert n_items == 2 ** n_modes, f"n_items ({n_items}) must equal 2**n_modes ({2**n_modes})"
    V = np.zeros((n_modes, n_items))
    for i in range(n_items):
        for alpha in range(n_modes):
            bit = (i >> (n_modes - 1 - alpha)) & 1
            V[alpha, i] = 1.0 if bit == 0 else -1.0
    V /= np.sqrt(n_items)
    return V


def run(cfg: Sim2Config, out_path: str = DEFAULT_OUT_PATH) -> str:
    torch.manual_seed(cfg.seed)
    n_modes = cfg.output_dim

    X = make_orthonormal_inputs(cfg.n_items, cfg.input_dim, seed=cfg.seed)  # (n_items, input_dim)
    ortho_err = (X @ X.T - torch.eye(cfg.n_items, dtype=X.dtype)).abs().max().item()
    assert ortho_err < 1e-5, f"inputs not orthonormal: max err {ortho_err}"

    V = hadamard_v_matrix(cfg.n_items, n_modes)  # (n_modes, n_items)
    s = np.array(cfg.singular_values, dtype=np.float64)
    assert s.shape == (n_modes,), f"singular_values must have length {n_modes}"
    Y_items = (V * s[:, None]).T  # (n_items, output_dim); row i = target vector for item i
    Y_items_t = torch.tensor(Y_items, dtype=torch.float64)

    # Small random Gaussian init (Saxe et al. 2019 PNAS, SI: "networks were
    # initialized with independent random Gaussian weights... W1(0)_ij ~
    # N(0, a0^2/N1)") -- NOT deliberately aligned to the task's SVD directions.
    # Fan-in normalized per layer (standard practice, avoids width-dependent
    # blow-up/vanishing) and scaled by a0^(1/L) per layer so the L-layer
    # composition's overall scale is ~a0. Approximate balance/alignment is
    # expected to emerge from smallness + randomness, not be built in.
    scale = cfg.a0 ** (1.0 / L)
    net = DeepLinearNet(cfg.input_dim, cfg.h1_dim, cfg.h2_dim, output_dim=cfg.output_dim, seed=cfg.seed)
    with torch.no_grad():
        g = torch.Generator().manual_seed(cfg.seed + 1)
        net.W1.weight.copy_(scale * torch.randn(net.W1.weight.shape, generator=g, dtype=torch.float64)
                             / (cfg.input_dim ** 0.5))
        net.W2.weight.copy_(scale * torch.randn(net.W2.weight.shape, generator=g, dtype=torch.float64)
                             / (cfg.h1_dim ** 0.5))
        net.W3.weight.copy_(scale * torch.randn(net.W3.weight.shape, generator=g, dtype=torch.float64)
                             / (cfg.h2_dim ** 0.5))

    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr)

    T = cfg.n_trials
    h1_pre = np.zeros((T, cfg.n_items, cfg.h1_dim), dtype=np.float64)
    h1_post = np.zeros_like(h1_pre)
    h2_pre = np.zeros((T, cfg.n_items, cfg.h2_dim), dtype=np.float64)
    h2_post = np.zeros_like(h2_pre)
    y_pre = np.zeros((T, cfg.n_items, cfg.output_dim), dtype=np.float64)
    y_post = np.zeros_like(y_pre)
    anchor_idx = np.zeros(T, dtype=np.int64)
    anchor_loss = np.zeros(T, dtype=np.float64)
    a_pre = np.zeros((T, n_modes), dtype=np.float64)  # learned mode amplitudes a_alpha(t)

    order_rng = np.random.default_rng(cfg.seed + 2)
    queue = []
    for trial in range(cfg.n_trials):
        if not queue:
            queue = list(order_rng.permutation(cfg.n_items))
        a = int(queue.pop(0))

        h1, h2, y = net(X)
        h1_pre[trial] = h1.detach().numpy()
        h2_pre[trial] = h2.detach().numpy()
        y_pre_np = y.detach().numpy()
        y_pre[trial] = y_pre_np
        for alpha in range(n_modes):
            a_pre[trial, alpha] = V[alpha] @ y_pre_np[:, alpha]

        # loss = 1/2 ||y_a - Y_a||^2, same convention as sim 1/2, generalized
        # to a vector output (sum of squared error across output dims).
        loss = 0.5 * ((y[a] - Y_items_t[a]) ** 2).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            h1_2, h2_2, y_2 = net(X)
        h1_post[trial] = h1_2.numpy()
        h2_post[trial] = h2_2.numpy()
        y_post[trial] = y_2.numpy()

        anchor_idx[trial] = a
        anchor_loss[trial] = loss.item()

        if (trial + 1) % max(1, cfg.n_trials // 10) == 0:
            window = max(1, cfg.n_items)
            recent = anchor_loss[max(0, trial + 1 - window):trial + 1].mean()
            print(f"trial {trial + 1}/{cfg.n_trials}  mean anchor loss (last {window}) = {recent:.5f}  "
                  f"a(t) = {np.round(a_pre[trial], 4)}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        h1_pre=h1_pre, h1_post=h1_post,
        h2_pre=h2_pre, h2_post=h2_post,
        y_pre=y_pre, y_post=y_post,
        anchor_idx=anchor_idx, anchor_loss=anchor_loss,
        a_pre=a_pre,
        X=X.numpy(), V=V, singular_values=s, Y_items=Y_items,
        W1_final=net.W1.weight.detach().numpy(),
        W2_final=net.W2.weight.detach().numpy(),
        W3_final=net.W3.weight.detach().numpy(),
        n_items=cfg.n_items, input_dim=cfg.input_dim,
        h1_dim=cfg.h1_dim, h2_dim=cfg.h2_dim, output_dim=cfg.output_dim,
        a0=cfg.a0, lr=cfg.lr, n_trials=cfg.n_trials, seed=cfg.seed,
    )
    print(f"Saved results to {out_path}")
    return out_path


if __name__ == "__main__":
    run(Sim2Config())
