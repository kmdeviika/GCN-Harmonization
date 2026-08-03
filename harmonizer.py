"""
GCN harmonizer --  implementation.

"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ---------------------------------------------------------------------------
# Reference electrode positions
# ---------------------------------------------------------------------------

def load_sphere2_reference_csv(csv_path: str):
    """
    Loads oostenveld_sphere2_59_reference_coordinates.csv 
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values("reference_index")
    order = df["reference_name"].tolist()
    positions = {
        row["reference_name"]: (row["x_m"], row["y_m"], row["z_m"])
        for _, row in df.iterrows()
    }
    return order, positions


# ---------------------------------------------------------------------------
# Adjacency construction 
# ---------------------------------------------------------------------------

def build_adjacency_from_cartesian(positions: dict, order: list,
                                    R: float = 10.0, sigma: float = 5.0,
                                    threshold: float = 0.001) -> np.ndarray:
    """
    Geodesic-distance Gaussian-weighted adjacency,
    Diagonal set to 1.0 (self-similarity, matching A~ = A + I) 
    """
    n = len(order)
    theta = np.zeros(n)
    phi = np.zeros(n)
    for i, name in enumerate(order):
        x, y, z = positions[name]
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2)
        theta[i] = np.arccos(z / r)
        phi[i] = np.arctan2(y, x)

    A = np.zeros((n, n), dtype=np.float64)
    for i in range(n):
        for j in range(n):
            if i == j:
                A[i, j] = 1.0
                continue
            cos_d = (np.sin(theta[i]) * np.sin(theta[j])
                     + np.cos(theta[i]) * np.cos(theta[j]) * np.cos(phi[i] - phi[j]))
            cos_d = np.clip(cos_d, -1.0, 1.0)
            d = R * np.arccos(cos_d)
            w = np.exp(-(d ** 2) / (2 * sigma ** 2))
            A[i, j] = w
            A[j, i] = w

    A[A < threshold] = 0.0
    return A


def dense_adjacency_to_edge_index(A: np.ndarray, device):
    
    
    A = A.copy()
    np.fill_diagonal(A, 0.0)
    rows, cols = np.nonzero(A)
    weights = A[rows, cols]
    edge_index = torch.tensor(np.stack([rows, cols]), dtype=torch.long, device=device)
    edge_weight = torch.tensor(weights, dtype=torch.float32, device=device)
    return edge_index, edge_weight


def build_batched_edges(edge_index: torch.Tensor, edge_weight: torch.Tensor,
                         num_graphs: int, num_nodes: int):
    
    device = edge_index.device
    n_edges = edge_index.shape[1]
    offsets = (torch.arange(num_graphs, device=device) * num_nodes).repeat_interleave(n_edges)
    batched_edge_index = edge_index.repeat(1, num_graphs) + offsets.unsqueeze(0)
    batched_edge_weight = edge_weight.repeat(num_graphs)
    return batched_edge_index, batched_edge_weight


class _BatchedEdgeCache:
    """Avoids rebuilding the batched edge_index/edge_weight every forward
    call -- only rebuilt when num_graphs (i.e. batch size) actually changes."""
    def __init__(self):
        self._cache = {}

    def get(self, edge_index, edge_weight, num_graphs, num_nodes):
        if num_graphs not in self._cache:
            self._cache[num_graphs] = build_batched_edges(edge_index, edge_weight, num_graphs, num_nodes)
        return self._cache[num_graphs]


# ---------------------------------------------------------------------------
# Model: GCNHarmonizerDense
# ---------------------------------------------------------------------------

class GCNHarmonizerDense(nn.Module):
    def __init__(self, n_channels: int, n_samples: int, hidden: int = 128):
        super().__init__()
        self.n_channels = n_channels
        self.n_samples = n_samples
        self.conv1 = GCNConv(n_samples, hidden, cached=False)
        self.conv2 = GCNConv(hidden, n_samples, cached=False)
        self._edge_cache = _BatchedEdgeCache()

    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        # X: (B, Nc, Ns) -- Ns MUST equal self.n_samples (fixed-length architecture)
        B, Nc, Ns = X.shape
        assert Ns == self.n_samples, (
            f"GCNHarmonizerDense trained for Ns={self.n_samples}, got Ns={Ns}."
        )
        batched_edge_index, batched_edge_weight = self._edge_cache.get(edge_index, edge_weight, B, Nc)
        x_flat = X.reshape(B * Nc, Ns)
        h1 = F.relu(self.conv1(x_flat, batched_edge_index, batched_edge_weight))
        h2 = self.conv2(h1, batched_edge_index, batched_edge_weight)
        return h2.reshape(B, Nc, Ns)


# ---------------------------------------------------------------------------
# Masking utilities
# ---------------------------------------------------------------------------

def curriculum_cap(step: int, warmup_steps: int, m_min: int, m_max: int) -> int:
    
    frac = min(step / max(warmup_steps, 1), 1.0)
    return int(round(m_min + (m_max - m_min) * frac))


def sample_batch_masks(batch_size: int, n_channels: int, m: int,
                        device: torch.device) -> torch.Tensor:
    """Random per-trial electrode masking. m is the number of MASKED
    (missing) electrodes; mask=1 means observed, mask=0 means missing."""
    masks = torch.ones(batch_size, n_channels, device=device)
    for b in range(batch_size):
        idx = torch.randperm(n_channels, device=device)[:m]
        masks[b, idx] = 0
    return masks


# ---------------------------------------------------------------------------
# Losses 
# ---------------------------------------------------------------------------

def masked_reconstruction_loss(X_true: torch.Tensor, X_pred: torch.Tensor,
                                mask: torch.Tensor, lambda_obs: float = 0.1) -> torch.Tensor:
    
    miss = (1 - mask).unsqueeze(-1)
    obs = mask.unsqueeze(-1)
    n_missing = miss.sum().clamp(min=1)
    n_obs = obs.sum().clamp(min=1)
    ns = X_true.shape[-1]
    l_missing = ((miss * (X_pred - X_true)) ** 2).sum() / (n_missing * ns)
    l_obs = ((obs * (X_pred - X_true)) ** 2).sum() / (n_obs * ns)
    return l_missing + lambda_obs * l_obs


def full_reconstruction_mse(X_true: torch.Tensor, X_pred: torch.Tensor) -> torch.Tensor:
    """
    Plain MSE over ALL electrodes .
    """
    return F.mse_loss(X_pred, X_true)


def real_preserving_output(X_true: torch.Tensor, X_pred: torch.Tensor,
                            mask: torch.Tensor) -> torch.Tensor:
    """Real, observed electrode values are NEVER replaced by the model's
    guess -- only genuinely missing electrodes get the reconstructed value."""
    m = mask.unsqueeze(-1)
    return m * X_true + (1 - m) * X_pred
