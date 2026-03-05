"""PyTorch Dataset and collation for prosthesis denoising training."""

from __future__ import annotations

import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from neuromap_sim.apps.prosthesis.config import GammatoneFeatureConfig
from neuromap_sim.apps.prosthesis.features import extract_gammatone_features
from neuromap_sim.apps.signal_processing.audio_io import load_audio_mono


def read_manifest_rows(manifest_path: Path | str) -> list[dict[str, str]]:
    """Read a prosthesis manifest CSV and return a list of row dicts."""
    manifest_path = Path(manifest_path)
    if not manifest_path.exists():
        return []
    with manifest_path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


@dataclass(slots=True)
class Batch:
    """A collated mini-batch of (input, target) feature tensors."""

    x: torch.Tensor
    y: torch.Tensor


class ProsthesisDenoisingDataset(Dataset):
    """Dataset that loads noisy features (.npy) and extracts clean targets on the fly.

    Parameters
    ----------
    rows:
        Manifest rows as returned by :func:`read_manifest_rows`.  Each row
        must contain ``sample_id``, ``feature_path``, and ``clean_path``.
    sample_rate:
        Audio sample rate used for clean-target feature extraction.
    feature_config:
        Gammatone feature configuration matching the stored feature tensors.
    target_cache_size:
        Number of clean target features to keep in an LRU cache.
    """

    def __init__(
        self,
        rows: list[dict[str, str]],
        *,
        sample_rate: int,
        feature_config: GammatoneFeatureConfig,
        target_cache_size: int = 256,
    ) -> None:
        self.rows = rows
        self.sample_rate = sample_rate
        self.feature_config = feature_config
        self.target_cache_size = max(1, target_cache_size)
        self._target_cache: OrderedDict[str, np.ndarray] = OrderedDict()

    def __len__(self) -> int:
        return len(self.rows)

    def _load_clean_target(self, row: dict[str, str]) -> np.ndarray:
        key = row["sample_id"]
        cached = self._target_cache.get(key)
        if cached is not None:
            self._target_cache.move_to_end(key)
            return cached

        clean_audio, _ = load_audio_mono(row["clean_path"], target_sample_rate=self.sample_rate)
        clean_features = extract_gammatone_features(
            clean_audio,
            sample_rate=self.sample_rate,
            config=self.feature_config,
        )
        target = clean_features.astype(np.float32)

        self._target_cache[key] = target
        if len(self._target_cache) > self.target_cache_size:
            self._target_cache.popitem(last=False)
        return target

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        row = self.rows[index]
        feature_path = Path(row["feature_path"])
        noisy_features = np.load(feature_path, allow_pickle=False).astype(np.float32)
        if noisy_features.ndim != 2:
            raise ValueError(
                f"Expected 2D feature tensor in {feature_path}, got {noisy_features.shape}."
            )
        clean_target = self._load_clean_target(row)
        # Keep noisy/clean frame lengths aligned for sequence regression.
        frame_count = min(noisy_features.shape[0], clean_target.shape[0])
        noisy_aligned = noisy_features[:frame_count, :]
        clean_aligned = clean_target[:frame_count, :]
        return torch.from_numpy(noisy_aligned), torch.from_numpy(clean_aligned)


def collate_batch(samples: list[tuple[torch.Tensor, torch.Tensor]]) -> Batch:
    """Collate variable-length feature pairs into a zero-padded :class:`Batch`."""
    features, targets = zip(*samples)
    max_steps = max(item.shape[0] for item in features)
    input_size = features[0].shape[1]
    x_batch = torch.zeros((len(features), max_steps, input_size), dtype=torch.float32)
    y_batch = torch.zeros((len(features), max_steps, input_size), dtype=torch.float32)
    for idx, (x_item, y_item) in enumerate(zip(features, targets, strict=True)):
        steps = min(x_item.shape[0], y_item.shape[0], max_steps)
        x_batch[idx, :steps, :] = x_item[:steps]
        y_batch[idx, :steps, :] = y_item[:steps]
    return Batch(x=x_batch, y=y_batch)

