
import os

import numpy as np
import torch
from scipy import signal as sp_signal

from eeg_utils import load_mat_data, bandpass_filtering
from harmonizer import (
    load_sphere2_reference_csv, build_adjacency_from_cartesian,
    dense_adjacency_to_edge_index,
    GCNHarmonizerDense,
    curriculum_cap, sample_batch_masks,
    masked_reconstruction_loss,
)


def load_and_check(cfg):
    X, Y = load_mat_data(cfg["data"]["mat_files"], cfg["data"]["data_dir"])
    print(f"Loaded X shape: {X.shape}  (trials, channels, samples)")
    print(f"Loaded Y shape: {Y.shape}")

    csv_path = os.path.join(cfg["data"]["data_dir"], cfg["data"]["sphere2_reference_csv"])
    order, positions = load_sphere2_reference_csv(csv_path)

    n_channels_data = X.shape[1]
    n_channels_csv = len(order)
    print(f"Channels in data                              : {n_channels_data}")
    print(f"Channels in oostenveld_sphere2_59_reference.csv: {n_channels_csv}")
    if n_channels_data != n_channels_csv:
        raise ValueError(
            f"MISMATCH: data has {n_channels_data} channels but "
            f"sphere2 reference has {n_channels_csv}. Resolve before proceeding."
        )
    return X, Y, order, positions


def preprocess(X, cfg, external_mean_std=None):
    """
    X: (trials, channels, samples). Returns (X_norm, crop_samples, mean, std).
    
    """
    current_sfreq = cfg["data"]["current_sfreq"]
    target_sfreq = cfg["data"]["target_sfreq"]
    crop_samples = int(target_sfreq * cfg["data"]["crop_seconds"])
    band = cfg["data"]["band"]
    filt_order = cfg["data"]["filt_order"]

    X_for_filter = np.transpose(X, (0, 2, 1))  # -> (trials, samples, channels)
    X_filt = bandpass_filtering(X_for_filter, fs=current_sfreq, fcut=band, filt_order=filt_order)

    n_trials, n_samples, n_channels = X_filt.shape
    new_len = int(round(n_samples * target_sfreq / current_sfreq))
    X_resampled = sp_signal.resample(X_filt, new_len, axis=1)

    if X_resampled.shape[1] < crop_samples:
        raise ValueError(
            f"Resampled length {X_resampled.shape[1]} < target crop "
            f"{crop_samples}. Check current_sfreq in config."
        )
    X_cropped = X_resampled[:, -crop_samples:, :]

    if external_mean_std is not None:
        mean, std = external_mean_std
    else:
        mean = X_cropped.mean(axis=(0, 1), keepdims=True)
        std = X_cropped.std(axis=(0, 1), keepdims=True) + 1e-8
    X_norm = (X_cropped - mean) / std
    X_norm = X_cropped
    

    return X_norm, crop_samples, mean, std


def train_variant(model, model_name, X_tensor, edge_index, edge_weight, device, cfg):
    model = model.to(device)
    tcfg = cfg["training"]
    mcfg = cfg["masking"]
    optimizer = torch.optim.Adam(model.parameters(), lr=tcfg["lr"])
    n_trials, n_channels, n_samples = X_tensor.shape
    step = 0

    print(f"\n{'=' * 50}\nTraining {model_name}\n{'=' * 50}")

    for epoch in range(tcfg["n_epochs"]):
        perm = torch.randperm(n_trials)
        epoch_loss, n_batches = 0.0, 0

        for i in range(0, n_trials, tcfg["batch_size"]):
            idx = perm[i:i + tcfg["batch_size"]]
            X_batch = X_tensor[idx].to(device)

            m_cap = curriculum_cap(step, mcfg["warmup_steps"], mcfg["m_min"], mcfg["m_max_target"])
            m = int(torch.randint(mcfg["m_min"], max(m_cap, mcfg["m_min"] + 1), (1,)).item())
            mask = sample_batch_masks(X_batch.shape[0], n_channels, m, device)

            X_masked = X_batch * mask.unsqueeze(-1)
            X_pred = model(X_masked, edge_index, edge_weight)
            loss = masked_reconstruction_loss(X_batch, X_pred, mask, mcfg["lambda_obs"])

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=tcfg["grad_clip_norm"])
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            step += 1

        if epoch % tcfg["log_every"] == 0 or epoch == tcfg["n_epochs"] - 1:
            print(f"  epoch {epoch:3d} | m_cap={m_cap:2d} | avg loss={epoch_loss / n_batches:.4f}")

    return model


def main(config_path):
    import yaml
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, Y, order, positions = load_and_check(cfg)
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

    os.makedirs(cfg["output"]["results_dir"], exist_ok=True)
    model = GCNHarmonizerDense(n_channels, crop_samples, hidden=cfg["model"]["hidden_dim"])
    model = train_variant(model, "GCNHarmonizerDense (curriculum)", X_tensor, edge_index, edge_weight, device, cfg)
    torch.save(model.state_dict(), os.path.join(cfg["output"]["results_dir"], "gcn_dense_curriculum.pt"))
    print(f"\nSaved checkpoint to {cfg['output']['results_dir']}.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
