from dataclasses import dataclass


@dataclass
class Sim3Config:
    n_items: int = 8               # must be 2**tree_depth
    tree_depth: int = 3            # D: number of hierarchy levels (binary branching generations)
    input_dim: int = 16
    h1_dim: int = 500               # widened (32->48->500) to try to reduce cross-mode leakage
    h2_dim: int = 500               # widened (32->48->500) to try to reduce cross-mode leakage
    # output_dim is derived: n_features (network regresses onto the raw sampled attribute vector)
    n_features: int = 200          # N3 in Saxe et al.'s notation -- number of independently-sampled
                                    # branching-diffusion attributes. Not stated by the paper for this
                                    # tree size (confirmed 2026-08-18: their only N3 value, 7, is for
                                    # an unrelated 4-item example and is rank-deficient for our 8-item
                                    # tree, which needs >= 8). Revisiting 200 after 2000/12/7 -- all
                                    # showed the same underlying same-level eigenvector degeneracy
                                    # (a genuine identifiability limit, not finite-sample noise; see
                                    # project memory), so no N3 choice fully "fixes" it.
    mutation_epsilon: float = 0.15  # per-branch flip probability; matches Saxe et al.'s own value
                                     # for their 3-level binary-branching tree example (SI Appendix)
    a0: float = 1e-4               # init scale (small random Gaussian, per level 2026-08-17 correction)
    lr: float = 0.01               # loss sums over n_features=200; lr=0.01 converges cleanly (was
                                    # 0.05 for n_features=12/7, too large now -- rescale with n_features)
    n_trials: int = 2000            # convergence confirmed by ~1800 at this lr/n_features
    seed: int = 0
