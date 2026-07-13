from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class RandomNodeMaskDataset(Dataset):
    """Randomly mask complete electrode time-series embeddings."""

    def __init__(
        self,
        x: np.ndarray,
        *,
        min_masked: int,
        max_masked: int,
        seed: int,
        deterministic: bool,
    ) -> None:
        if min_masked < 1:
            raise ValueError("min_masked must be at least 1.")
        if max_masked < min_masked:
            raise ValueError("max_masked must be >= min_masked.")
        if max_masked >= x.shape[1]:
            raise ValueError("At least one electrode must remain observed.")

        self.x = torch.tensor(x, dtype=torch.float32)
        self.min_masked = int(min_masked)
        self.max_masked = int(max_masked)
        self.seed = int(seed)
        self.deterministic = bool(deterministic)
        self.rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, index: int):
        rng = (
            np.random.default_rng(self.seed + int(index))
            if self.deterministic
            else self.rng
        )
        number_masked = int(
            rng.integers(
                self.min_masked,
                self.max_masked + 1,
            )
        )
        missing = rng.choice(
            self.x.shape[1],
            size=number_masked,
            replace=False,
        )

        observed = torch.ones(
            self.x.shape[1],
            dtype=torch.bool,
        )
        observed[missing] = False

        x_input = self.x[index].clone()
        x_input[~observed] = 0.0

        return x_input, self.x[index], observed, index
