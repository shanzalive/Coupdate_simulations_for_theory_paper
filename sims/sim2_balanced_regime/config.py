from dataclasses import dataclass


@dataclass
class Sim2Config:
    n_items: int = 8          # must be 2**n_modes for the Hadamard hierarchy below
    input_dim: int = 16
    h1_dim: int = 32
    h2_dim: int = 16
    output_dim: int = 3       # = n_modes (r): one output slot per hierarchy level
    singular_values: tuple = (3.0, 2.0, 1.0)  # s_alpha, largest first
    a0: float = 1e-4          # shared small initial mode amplitude (balanced init)
    lr: float = 0.05          # same as sim 1/2
    n_trials: int = 6000      # single-item trials; tuned so all 3 modes rise and separate
    seed: int = 0
