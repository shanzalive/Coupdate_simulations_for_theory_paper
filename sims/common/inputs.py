"""Orthogonal input construction, shared across simulations in this paper."""
import torch


def make_orthonormal_inputs(n_items: int, dim: int, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """Return an (n_items, dim) tensor of mutually orthonormal rows.

    Vectors are random orthonormal directions (via QR of a random Gaussian
    matrix), not axis-aligned one-hot codes — the layer-1 zero-co-update
    result should follow from orthogonality alone, not from localist coding.

    float64 by default: the layer-1 shift is analytically exact zero, and
    float32 rounding noise (~1e-7) is otherwise large enough to produce
    spurious nonzero correlations in downstream co-update analyses.
    """
    if n_items > dim:
        raise ValueError(f"n_items ({n_items}) must be <= dim ({dim}) for orthonormal inputs")
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(dim, dim, generator=g, dtype=dtype)
    Q, _ = torch.linalg.qr(A)
    X = Q[:, :n_items].T.contiguous()  # (n_items, dim)
    return X


def make_correlated_inputs(n_items: int, dim: int, rho: float, seed: int = 0, dtype=torch.float64) -> torch.Tensor:
    """Return an (n_items, dim) tensor of unit vectors with equal pairwise
    correlation `rho` for every pair (an equicorrelated set), not a random
    per-pair correlation.

    Construction: draw n_items+1 orthonormal vectors u, e_1..e_n (via
    make_orthonormal_inputs), then set x_i = sqrt(rho)*u + sqrt(1-rho)*e_i.
    Since every e_i is orthonormal to every e_j and to u, the cross terms
    vanish and x_i . x_j = rho exactly for all i != j, with ||x_i|| = 1.
    """
    if not (0.0 <= rho < 1.0):
        raise ValueError(f"rho must be in [0, 1): got {rho}")
    basis = make_orthonormal_inputs(n_items + 1, dim, seed=seed, dtype=dtype)  # (n_items+1, dim)
    u = basis[0]
    e = basis[1:]  # (n_items, dim)
    X = (rho ** 0.5) * u.unsqueeze(0) + ((1 - rho) ** 0.5) * e
    return X
