"""Train the depth-3 deep linear network on a single-anchor-per-trial curriculum.

Each trial picks one "anchor" item, trains on it (feedback + SGD step), and
records the hidden representations of ALL items (anchor + probes), at every
layer, both immediately before and immediately after that single update.

Items are drawn from a shuffled queue that's refilled with a fresh random
permutation of all n_items whenever it runs out -- this guarantees every
item appears once before any item repeats, without exposing a separate
named "pass"/"epoch" concept: trial is the only unit.
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
import torch

from sims.common.inputs import make_correlated_inputs, make_orthonormal_inputs
from sims.common.linear_net import DeepLinearNet
from sims.sim1_layerwise_coupdate.config import Sim1Config

DEFAULT_OUT_PATH_ORTHOGONAL = os.path.join(
    _PROJECT_ROOT, "outputs", "sim1_layerwise_coupdate", "results_orthogonal.npz"
)
DEFAULT_OUT_PATH_CORRELATED = os.path.join(
    _PROJECT_ROOT, "outputs", "sim1_layerwise_coupdate", "results_correlated.npz"
)


def run(cfg: Sim1Config, out_path: str) -> str:
    torch.manual_seed(cfg.seed)

    if cfg.rho is None:
        X = make_orthonormal_inputs(cfg.n_items, cfg.input_dim, seed=cfg.seed)
        ortho_err = (X @ X.T - torch.eye(cfg.n_items, dtype=X.dtype)).abs().max().item()
        assert ortho_err < 1e-5, f"inputs not orthonormal: max err {ortho_err}"
    else:
        X = make_correlated_inputs(cfg.n_items, cfg.input_dim, rho=cfg.rho, seed=cfg.seed)
        target_gram = cfg.rho * torch.ones(cfg.n_items, cfg.n_items, dtype=X.dtype)
        target_gram.fill_diagonal_(1.0)
        corr_err = (X @ X.T - target_gram).abs().max().item()
        assert corr_err < 1e-5, f"inputs not at target correlation: max err {corr_err}"

    n_pos = cfg.n_items // 2
    # Deterministic, not shuffled: first n_pos items get class_values[0] (+1),
    # remaining items get class_values[1] (-1) -- so item order already
    # reflects label order (used to organize figure 2's C^(a)_2 heatmap).
    labels = np.array(
        [cfg.class_values[0]] * n_pos + [cfg.class_values[1]] * (cfg.n_items - n_pos)
    )
    v = torch.tensor(labels, dtype=torch.float64)

    net = DeepLinearNet(cfg.input_dim, cfg.h1_dim, cfg.h2_dim, init_std=cfg.init_std, seed=cfg.seed)
    optimizer = torch.optim.SGD(net.parameters(), lr=cfg.lr)

    T = cfg.n_trials
    h1_pre = np.zeros((T, cfg.n_items, cfg.h1_dim), dtype=np.float64)
    h1_post = np.zeros_like(h1_pre)
    h2_pre = np.zeros((T, cfg.n_items, cfg.h2_dim), dtype=np.float64)
    h2_post = np.zeros_like(h2_pre)
    y_pre = np.zeros((T, cfg.n_items), dtype=np.float64)
    y_post = np.zeros_like(y_pre)
    anchor_idx = np.zeros(T, dtype=np.int64)
    anchor_loss = np.zeros(T, dtype=np.float64)
    W3_pre = np.zeros((T, cfg.h2_dim), dtype=np.float64)

    order_rng = np.random.default_rng(cfg.seed + 2)
    queue = []
    for trial in range(cfg.n_trials):
        if not queue:
            queue = list(order_rng.permutation(cfg.n_items))
        a = int(queue.pop(0))

        h1, h2, y = net(X)
        h1_pre[trial] = h1.detach().numpy()
        h2_pre[trial] = h2.detach().numpy()
        y_pre[trial] = y.detach().numpy().squeeze(-1)
        W3_pre[trial] = net.W3.weight.detach().numpy().squeeze(0)

        # loss = 1/2 (y_a - v_a)^2, matching the paper's convention
        # ell(f,y) = 1/2||f-y||^2, r_a := v_a - y_a, so that the SGD
        # update to W2/W3 exactly matches eq. 16 (no extra factor of 2).
        loss = 0.5 * (y[a].squeeze() - v[a]) ** 2
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            h1_2, h2_2, y_2 = net(X)
        h1_post[trial] = h1_2.numpy()
        h2_post[trial] = h2_2.numpy()
        y_post[trial] = y_2.numpy().squeeze(-1)

        anchor_idx[trial] = a
        anchor_loss[trial] = loss.item()

        if (trial + 1) % max(1, cfg.n_trials // 10) == 0:
            window = max(1, cfg.n_items)
            recent = anchor_loss[max(0, trial + 1 - window):trial + 1].mean()
            print(f"trial {trial + 1}/{cfg.n_trials}  mean anchor loss (last {window}) = {recent:.5f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.savez_compressed(
        out_path,
        h1_pre=h1_pre, h1_post=h1_post,
        h2_pre=h2_pre, h2_post=h2_post,
        y_pre=y_pre, y_post=y_post,
        anchor_idx=anchor_idx, anchor_loss=anchor_loss,
        W3_pre=W3_pre,
        X=X.numpy(), v=v.numpy(),
        W1_final=net.W1.weight.detach().numpy(),
        W2_final=net.W2.weight.detach().numpy(),
        W3_final=net.W3.weight.detach().numpy(),
        n_items=cfg.n_items, input_dim=cfg.input_dim,
        h1_dim=cfg.h1_dim, h2_dim=cfg.h2_dim, lr=cfg.lr,
        n_trials=cfg.n_trials, seed=cfg.seed,
        rho=cfg.rho if cfg.rho is not None else np.nan,
    )
    print(f"Saved results to {out_path}")
    return out_path


if __name__ == "__main__":
    print("=== Orthogonal-input condition ===")
    run(Sim1Config(rho=None), out_path=DEFAULT_OUT_PATH_ORTHOGONAL)
    print("\n=== Correlated-input condition (rho=0.2) ===")
    run(Sim1Config(rho=0.2), out_path=DEFAULT_OUT_PATH_CORRELATED)
