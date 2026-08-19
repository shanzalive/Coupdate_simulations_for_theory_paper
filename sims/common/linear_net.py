"""Deep linear network, shared across simulations in this paper."""
import torch
import torch.nn as nn


class DeepLinearNet(nn.Module):
    """Depth-3 deep linear network: input -> h1 -> h2 -> output.

    No biases, no nonlinearities — every layer is a pure linear map, so
    the composed function is y = W3 @ W2 @ W1 @ x.

    output_dim defaults to 1 (sim 1/2's scalar-output network); sim 3 passes
    output_dim=r to get a multi-dimensional output, then overwrites the
    randomly-initialized weights below with a balanced construction.
    """

    def __init__(self, input_dim: int, h1_dim: int, h2_dim: int, output_dim: int = 1, init_std: float = 0.1,
                 seed: int = 0, dtype=torch.float64):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.W1 = nn.Linear(input_dim, h1_dim, bias=False, dtype=dtype)
        self.W2 = nn.Linear(h1_dim, h2_dim, bias=False, dtype=dtype)
        self.W3 = nn.Linear(h2_dim, output_dim, bias=False, dtype=dtype)
        for layer in (self.W1, self.W2, self.W3):
            layer.weight.data = torch.randn(layer.weight.shape, generator=g, dtype=dtype) * init_std

    def forward(self, x: torch.Tensor):
        h1 = self.W1(x)
        h2 = self.W2(h1)
        y = self.W3(h2)
        return h1, h2, y
