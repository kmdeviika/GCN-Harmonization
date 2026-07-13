from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


def build_geodesic_adjacency(
    electrode_csv: str | Path,
    *,
    head_radius_cm: float = 10.0,
    gaussian_sigma_cm: float = 5.0,
    threshold: float = 0.001,
    add_self_loops: bool = True,
) -> tuple[list[str], torch.Tensor]:
    table = pd.read_csv(electrode_csv)

    required = {"electrode_name", "xpos", "ypos", "zpos"}
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f"Electrode CSV missing columns: {sorted(missing)}"
        )

    names = [
        str(value).strip().strip("'").strip('"')
        for value in table["electrode_name"]
    ]
    if len({name.lower() for name in names}) != len(names):
        raise ValueError("Duplicate electrode names in electrode CSV.")

    xyz = table[["xpos", "ypos", "zpos"]].to_numpy(dtype=np.float64)
    if not np.isfinite(xyz).all():
        raise ValueError("Electrode coordinates contain NaN or Inf.")

    radius = np.linalg.norm(xyz, axis=1, keepdims=True)
    if np.any(radius <= 1e-12):
        raise ValueError("Invalid zero-length electrode coordinate.")

    unit = xyz / radius
    cosine = np.clip(unit @ unit.T, -1.0, 1.0)
    distance = head_radius_cm * np.arccos(cosine)

    adjacency = np.exp(
        -(distance ** 2) / (2.0 * gaussian_sigma_cm ** 2)
    )
    adjacency[adjacency < threshold] = 0.0

    if add_self_loops:
        np.fill_diagonal(adjacency, 1.0)
    else:
        np.fill_diagonal(adjacency, 0.0)

    adjacency = (adjacency + adjacency.T) / 2.0

    degree = adjacency.sum(axis=1)
    inverse_sqrt = np.power(
        np.clip(degree, 1e-12, None),
        -0.5,
    )
    normalized = (
        inverse_sqrt[:, None]
        * adjacency
        * inverse_sqrt[None, :]
    )

    return names, torch.tensor(normalized, dtype=torch.float32)
