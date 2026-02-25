"""Configuration dataclasses for prosthesis data bootstrap and generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(slots=True)
class GammatoneFeatureConfig:
    sample_rate: int = 16_000
    n_filters: int = 32
    low_freq_hz: float = 80.0
    high_freq_hz: float = 7_600.0
    frame_size: int = 256
    hop_size: int = 128
    compression: str = "log1p"
    power: float = 0.3

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProsthesisBootstrapConfig:
    raw_root: str = "data/prosthesis/raw"
    clean_limit: int | None = None
    noise_limit: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ProsthesisDatasetConfig:
    sample_rate: int = 16_000
    clip_seconds: float = 4.0
    min_duration_seconds: float = 0.3
    snr_min_db: float = 0.0
    snr_max_db: float = 20.0
    snr_bins: int = 5
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    target_peak: float = 0.9
    seed: int = 42
    max_pairs: int | None = None
    noise_metadata_csv: str | None = None
    features: GammatoneFeatureConfig = field(default_factory=GammatoneFeatureConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["features"] = self.features.to_dict()
        return payload

