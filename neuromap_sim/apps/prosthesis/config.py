"""Configuration dataclasses for prosthesis data bootstrap, generation, and training."""

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


@dataclass(slots=True)
class ProsthesisTrainConfig:
    """Configuration for training a prosthesis SNN via the SDK."""

    # Network
    hidden_size: int = 128
    use_decoder: bool = True
    chip_name: str | None = None  # None = auto-detect topology from data

    # Training
    epochs: int = 40
    batch_size: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    grad_clip_norm: float = 1.0
    scheduler: str = "plateau"
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    min_learning_rate: float = 1e-6
    lr_warmup_epochs: int = 3
    early_stop_patience: int = 8
    early_stop_min_delta: float = 1e-4
    loss_alpha: float = 0.8

    # Data
    feature_sample_rate: int = 16_000
    max_train_rows: int = 0
    max_val_rows: int = 0
    num_workers: int = 0
    target_cache_size: int = 256
    seed: int = 42
    device: str = ""

    # Feature extraction (for clean target recomputation)
    features: GammatoneFeatureConfig = field(default_factory=GammatoneFeatureConfig)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["features"] = self.features.to_dict()
        return payload


@dataclass(slots=True)
class ProsthesisExportConfig:
    """Configuration for exporting a trained prosthesis model."""

    bits: int = 4
    chip_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

