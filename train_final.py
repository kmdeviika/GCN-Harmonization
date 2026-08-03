"""
Final full training -- matches the paper's actual deployment protocol
(Section 4.2): FIXED m=15 every trial (NOT curriculum/ramped), PLAIN MSE
over ALL 59 electrodes (NOT the missing-focused weighted loss used for
LOSO/Table-I), trained on ALL 7 subjects. Produces the deployed pretrained
checkpoint -- NOT the same checkpoint as train_harmonizer.py/loso_evaluate.py,
which serve a different purpose (masking-sweep validation).
"""

import os

import numpy as np
import torch

from train_harmonizer import load_and_check, preprocess
from harmonizer import (
    build_adjacency_from_cartesian, dense_adjacency_to_edge_index,
    GCNHarmonizerDense, sample_batch_masks, full_reconstruction_mse,
)


def train_final(model, X_tensor, edge_index, edge_weight, device, cfg):
    model = model.to(device)
    tcfg = cfg["training"]
    m_fixed = cfg["masking"]["m_fixed"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    n_trials, n_channels, n_samples = X_tensor.shape

    print(f"\n{'=' * 50}\n"
          f"Final training: GCNHarmonizerDense (fixed m={m_fixed}, full MSE)\n"
          f"{'=' * 50}")

    for epoch in range(tcfg["n_epochs"]):
        model.train()
        perm = torch.randperm(n_trials)
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, n_trials, tcfg["batch_size"]):
            idx = perm[i:i + tcfg["batch_size"]]
            X_batch = X_tensor[idx].to(device)

            mask = sample_batch_masks(X_batch.shape[0], n_channels, m_fixed, device)
            X_masked = X_batch * mask.unsqueeze(-1)
            X_pred = model(X_masked, edge_index, edge_weight)

            loss = full_reconstruction_mse(X_batch, X_pred)  # plain MSE, ALL electrodes

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tcfg["grad_clip_norm"])
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1

        if epoch % tcfg["log_every"] == 0 or epoch == tcfg["n_epochs"] - 1:
            print(f"  epoch {epoch:3d} | avg loss={epoch_loss / n_batches:.4f}")

    return model


def main(config_path):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, Y, order, positions = load_and_check(cfg)  # all 7 subjects, no held-out split
    X_prep, crop_samples, mean, std = preprocess(X, cfg)
    X_prep = np.transpose(X_prep, (0, 2, 1))
    X_tensor = torch.tensor(X_prep, dtype=torch.float32)

    A = build_adjacency_from_cartesian(
        positions, order,
        R=cfg["adjacency"]["R"], sigma=cfg["adjacency"]["sigma"],
        threshold=cfg["adjacency"]["threshold"],
    )
    edge_index, edge_weight = dense_adjacency_to_edge_index(A, device)
    n_channels = X_tensor.shape[1]

    model = GCNHarmonizerDense(n_channels, crop_samples, hidden=cfg["model"]["hidden_dim"])
    model = train_final(model, X_tensor, edge_index, edge_weight, device, cfg)

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    ckpt_path = os.path.join(cfg["output"]["results_dir"], "gcn_dense_final_pretrained.pt")
    torch.save(model.state_dict(), ckpt_path)
    np.savez(os.path.join(cfg["output"]["results_dir"], "norm_stats.npz"), mean=mean, std=std)

    print(f"\nFinal pretrained checkpoint saved to {ckpt_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
