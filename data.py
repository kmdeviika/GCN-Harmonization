from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import scipy.io
from scipy.signal import resample


CHANNEL_KEYS = (
    "channel_names",
    "channels",
    "ch_names",
    "chanlabels",
    "chan_names",
    "electrode_names",
    "electrodes",
)


@dataclass
class EEGDataset:
    x: np.ndarray
    subjects: np.ndarray
    channel_names: list[str]
    labels: np.ndarray | None = None


def _clean_name(value: object) -> str:
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()
        else:
            value = "".join(str(item) for item in value.reshape(-1).tolist())
    return str(value).strip().strip("'").strip('"')


def _validate_channels(channels: Sequence[str], context: str) -> list[str]:
    cleaned = [_clean_name(name) for name in channels]
    if not cleaned:
        raise ValueError(f"{context}: channel list is empty.")
    if len({name.lower() for name in cleaned}) != len(cleaned):
        raise ValueError(f"{context}: duplicate channel names detected.")
    return cleaned


def read_channel_order(path: str | Path) -> list[str]:
    channels = [
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    return _validate_channels(channels, str(path))


def read_reference_channels(electrode_csv: str | Path) -> list[str]:
    table = pd.read_csv(electrode_csv)
    if "electrode_name" not in table.columns:
        raise ValueError("Reference electrode CSV must contain 'electrode_name'.")
    return _validate_channels(table["electrode_name"].tolist(), str(electrode_csv))


def _mat_strings(value: np.ndarray) -> list[str]:
    array = np.asarray(value)
    if array.dtype.kind in {"U", "S"}:
        if array.ndim == 2 and array.shape[0] > 1 and array.shape[1] > 1:
            return ["".join(row.astype(str).tolist()).strip() for row in array]
        return [_clean_name(item) for item in array.reshape(-1)]
    if array.dtype == object:
        return [_clean_name(item) for item in array.reshape(-1)]
    return []


def extract_embedded_channels(mat: dict) -> list[str] | None:
    for key in CHANNEL_KEYS:
        if key in mat:
            channels = [name for name in _mat_strings(mat[key]) if name]
            if channels:
                return _validate_channels(channels, key)
    return None


def _to_trials_channels_samples(
    x: np.ndarray,
    expected_channels: int,
) -> np.ndarray:
    array = np.asarray(x)
    if array.ndim != 3:
        raise ValueError(f"Expected 3-D EEG data, found {array.shape}.")
    channel_axes = [
        axis for axis, size in enumerate(array.shape)
        if size == expected_channels
    ]
    if len(channel_axes) != 1:
        raise ValueError(
            f"Could not uniquely identify the {expected_channels}-channel axis "
            f"in shape {array.shape}."
        )
    channel_axis = channel_axes[0]
    remaining = [axis for axis in range(3) if axis != channel_axis]
    trial_axis = min(remaining, key=lambda axis: array.shape[axis])
    sample_axis = next(axis for axis in remaining if axis != trial_axis)
    return np.transpose(
        array,
        (trial_axis, channel_axis, sample_axis),
    ).astype(np.float32)


def load_mat_directory(
    directory: str | Path,
    *,
    eeg_key: str,
    channel_order_file: str | Path | None,
    labels_key: str | None = None,
    file_pattern: str = "*.mat",
    num_samples: int | None = None,
) -> EEGDataset:
    directory = Path(directory)
    paths = sorted(directory.glob(file_pattern))
    if not paths:
        raise FileNotFoundError(
            f"No files matching {file_pattern!r} found in {directory}."
        )

    external_channels = (
        read_channel_order(channel_order_file)
        if channel_order_file
        else None
    )

    all_x: list[np.ndarray] = []
    all_subjects: list[str] = []
    all_labels: list[np.ndarray] = []
    resolved_channels: list[str] | None = None

    for path in paths:
        mat = scipy.io.loadmat(path)
        if eeg_key not in mat:
            raise KeyError(f"{path.name}: missing EEG variable {eeg_key!r}.")

        channels = extract_embedded_channels(mat) or external_channels
        if channels is None:
            raise ValueError(
                f"{path.name}: no embedded channel labels and no "
                "channel_order_file was supplied."
            )

        x = _to_trials_channels_samples(
            mat[eeg_key],
            expected_channels=len(channels),
        )
        if num_samples is not None and x.shape[-1] != num_samples:
            x = resample(x, num_samples, axis=-1).astype(np.float32)

        if not np.isfinite(x).all():
            raise ValueError(f"{path.name}: EEG contains NaN or Inf.")

        if resolved_channels is None:
            resolved_channels = list(channels)
        elif [name.lower() for name in resolved_channels] != [
            name.lower() for name in channels
        ]:
            raise ValueError(
                f"{path.name}: channel order differs from earlier files."
            )

        all_x.append(x)
        all_subjects.extend([path.stem] * len(x))

        if labels_key is not None:
            if labels_key not in mat:
                raise KeyError(
                    f"{path.name}: missing label variable {labels_key!r}."
                )
            labels = np.asarray(mat[labels_key]).reshape(-1)
            if len(labels) != len(x):
                raise ValueError(
                    f"{path.name}: {len(x)} EEG trials but "
                    f"{len(labels)} labels."
                )
            all_labels.append(labels)

    assert resolved_channels is not None
    return EEGDataset(
        x=np.concatenate(all_x, axis=0),
        subjects=np.asarray(all_subjects, dtype=str),
        channel_names=resolved_channels,
        labels=np.concatenate(all_labels) if all_labels else None,
    )


def align_to_reference(
    x: np.ndarray,
    input_channels: Sequence[str],
    reference_channels: Sequence[str],
) -> tuple[np.ndarray, np.ndarray]:
    reference_index = {
        _clean_name(name).lower(): index
        for index, name in enumerate(reference_channels)
    }

    aligned = np.zeros(
        (x.shape[0], len(reference_channels), x.shape[2]),
        dtype=np.float32,
    )
    observed = np.zeros(len(reference_channels), dtype=bool)
    unmatched: list[str] = []

    for source_index, name in enumerate(input_channels):
        key = _clean_name(name).lower()
        if key not in reference_index:
            unmatched.append(str(name))
            continue
        target_index = reference_index[key]
        aligned[:, target_index, :] = x[:, source_index, :]
        observed[target_index] = True

    if unmatched:
        raise ValueError(
            f"Input channels absent from reference montage: {unmatched}."
        )
    if not observed.any():
        raise ValueError("No input channels overlap the reference montage.")

    return aligned, observed


def save_harmonized_npz(
    path: str | Path,
    *,
    harmonized: np.ndarray,
    aligned_input: np.ndarray,
    observed_mask: np.ndarray,
    subjects: np.ndarray,
    channel_names: Sequence[str],
    labels: np.ndarray | None,
) -> None:
    payload = {
        "harmonized_eeg": harmonized.astype(np.float32),
        "aligned_input_eeg": aligned_input.astype(np.float32),
        "observed_mask": observed_mask.astype(bool),
        "subjects": np.asarray(subjects, dtype=str),
        "channel_names": np.asarray(channel_names, dtype=str),
    }
    if labels is not None:
        payload["labels"] = labels

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(target, **payload)
