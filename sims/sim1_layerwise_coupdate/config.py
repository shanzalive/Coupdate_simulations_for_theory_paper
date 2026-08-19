from dataclasses import dataclass
from typing import Optional


@dataclass
class Sim1Config:
    n_items: int = 8
    input_dim: int = 16
    h1_dim: int = 32
    h2_dim: int = 16
    init_std: float = 0.1
    lr: float = 0.05
    n_trials: int = 240  # total single-item trials (one feedback + weight update each) = 30 passes of n_items=8
    seed: int = 0
    class_values: tuple = (1.0, -1.0)  # balanced binary value fn: half the items get class_values[0], half get class_values[1]
    rho: Optional[float] = None  # None = orthogonal inputs; float = equicorrelated inputs at this pairwise correlation
