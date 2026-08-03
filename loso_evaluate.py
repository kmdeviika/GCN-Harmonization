"""
Leave-One-Subject-Out (LOSO) evaluation of GCNHarmonizerDense
"""

import copy
import os

import numpy as np
import torch
import yaml

from train_harmonizer import load_and_check, preprocess, train_variant
from harmonizer import (
    build_adjacency_from_cartesian, dense_adjacency_to_edge_index,
    GCNHarmonizerDense, sample_batch_masks,
)


def evaluate_reconstruction_mse(model, X_holdout, edge_index, edge_weight, device,
                                 masking_levels, n_repeats=30):
    model.eval()
    n_trials, n_channels, n_samples = X_holdout.shape
    X_holdout = X_holdout.to(device)
    results = {}

    with torch.no_grad():
        for m in masking_levels:
            mse_accum = 0.0
            for _ in range(n_repeats):
                mask = sample_batch_masks(n_trials, n_channels, m, device)
                X_masked = X_holdout * mask.unsqueeze(-1)
                X_pred = model(X_masked, edge_index, edge_weight)
                miss = (1 - mask).unsqueeze(-1)
                sq_err = ((X_pred - X_holdout) ** 2) * miss
                mse = sq_err.sum() / (miss.sum() * n_samples)
                mse_accum += mse.item()
            results[m] = mse_accum / n_repeats

    return results


def main(config_path):
    with open(config_path) as f:
        base_cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    all_subjects = base_cfg["data"]["mat_files"]  # must be ALL 7 subjects for LOSO
    eval_levels = base_cfg["loso"]["eval_masking_levels"]
    n_repeats = base_cfg["loso"]["n_mask_repeats"]

    fold_results = {m: [] for m in eval_levels}

    for fold_idx, held_out_subject in enumerate(all_subjects):
        print(f"\n{'#' * 60}\nFOLD {fold_idx + 1}/{len(all_subjects)} "
              f"-- held out: {held_out_subject}\n{'#' * 60}")

        train_files = [f for f in all_subjects if f != held_out_subject]

        train_cfg = copy.deepcopy(base_cfg)
        train_cfg["data"]["mat_files"] = train_files
        X_train, Y_train, order, positions = load_and_check(train_cfg)
        X_train_prep, crop_samples, train_mean, train_std = preprocess(X_train, train_cfg)
        X_train_prep = np.transpose(X_train_prep, (0, 2, 1))
        X_train_tensor = torch.tensor(X_train_prep, dtype=torch.float32)

        holdout_cfg = copy.deepcopy(base_cfg)
        holdout_cfg["data"]["mat_files"] = [held_out_subject]
        X_hold, Y_hold, _, _ = load_and_check(holdout_cfg)
        X_hold_prep, _, _, _ = preprocess(X_hold, holdout_cfg, external_mean_std=(train_mean, train_std))
        X_hold_prep = np.transpose(X_hold_prep, (0, 2, 1))
        X_hold_tensor = torch.tensor(X_hold_prep, dtype=torch.float32)

        A = build_adjacency_from_cartesian(
            positions, order,
            R=base_cfg["adjacency"]["R"], sigma=base_cfg["adjacency"]["sigma"],
            threshold=base_cfg["adjacency"]["threshold"],
        )
        edge_index, edge_weight = dense_adjacency_to_edge_index(A, device)
        n_channels = X_train_tensor.shape[1]

        fold_dir = os.path.join(
            base_cfg["output"]["results_dir"],
            f"fold_{held_out_subject.replace('.mat', '')}",
        )
        os.makedirs(fold_dir, exist_ok=True)

        model = GCNHarmonizerDense(n_channels, crop_samples, hidden=base_cfg["model"]["hidden_dim"])
        model = train_variant(
            model, f"GCNHarmonizerDense (fold {held_out_subject})",
            X_train_tensor, edge_index, edge_weight, device, base_cfg,
        )
        torch.save(model.state_dict(), os.path.join(fold_dir, "gcn_dense.pt"))

        mse_fold = evaluate_reconstruction_mse(
            model, X_hold_tensor, edge_index, edge_weight, device, eval_levels, n_repeats
        )
        for m in eval_levels:
            fold_results[m].append(mse_fold[m])
        print(f"  MSE (fold {held_out_subject}): {mse_fold}")

    print(f"\n{'=' * 60}\nLOSO SUMMARY (mean +/- std across {len(all_subjects)} folds)\n{'=' * 60}")
    print(f"{'m':>4} | {'Dense (mean +/- std)':>24}")
    summary = {}
    for m in eval_levels:
        vals = np.array(fold_results[m])
        summary[m] = {"mean": float(vals.mean()), "std": float(vals.std())}
        print(f"{m:4d} | {vals.mean():10.4f} +/- {vals.std():8.4f}")

    os.makedirs(base_cfg["output"]["results_dir"], exist_ok=True)
    summary_path = os.path.join(base_cfg["output"]["results_dir"], "loso_summary.yaml")
    with open(summary_path, "w") as f:
        yaml.dump(summary, f)
    print(f"\nSaved summary to {summary_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()
    main(args.config)
