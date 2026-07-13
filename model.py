from __future__ import annotations

import torch
from torch import nn


class GCNHarmonizer(nn.Module):
    """
    Two-layer GCN matching the manuscript formulation.

    Shapes:
        H^(0) = X : [batch, N_c, N_s]
        H^(1)     : [batch, N_c, 128]
        H^(2)     : [batch, N_c, N_s]

    The number of graph nodes N_c remains unchanged. The trainable feature
    transformations are N_s -> 128 -> N_s.
    """

    def __init__(
        self,
        *,
        num_samples: int,
        hidden_features: int = 128,
    ) -> None:
        super().__init__()
        self.num_samples = int(num_samples)
        self.hidden_features = int(hidden_features)

        self.weight_1 = nn.Linear(
            self.num_samples,
            self.hidden_features,
            bias=True,
        )
        self.weight_2 = nn.Linear(
            self.hidden_features,
            self.num_samples,
            bias=True,
        )

    def forward(
        self,
        x: torch.Tensor,
        normalized_adjacency: torch.Tensor,
    ) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError(
                f"Expected [batch, electrodes, samples], found {tuple(x.shape)}."
            )
        if x.shape[-1] != self.num_samples:
            raise ValueError(
                f"Expected {self.num_samples} samples, found {x.shape[-1]}."
            )

        h = torch.einsum(
            "ij,bjf->bif",
            normalized_adjacency,
            x,
        )
        h = torch.relu(self.weight_1(h))

        h = torch.einsum(
            "ij,bjf->bif",
            normalized_adjacency,
            h,
        )
        return self.weight_2(h)
