"""Feature extraction and spike encoding for audio frames."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import torch


@dataclass(slots=True)
class SpikeEncodingConfig:
    frame_size: int = 512
    hop_size: int = 256
    input_size: int = 400
    threshold_scale: float = 0.9

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def _resample_feature_axis(features: np.ndarray, target_size: int) -> np.ndarray:
    source_size = features.shape[1]
    if source_size == target_size:
        return features
    source_idx = np.linspace(0.0, 1.0, source_size)
    target_idx = np.linspace(0.0, 1.0, target_size)
    out = np.empty((features.shape[0], target_size), dtype=np.float32)
    for i in range(features.shape[0]):
        out[i] = np.interp(target_idx, source_idx, features[i])
    return out


def extract_frame_features(frames: np.ndarray, *, input_size: int) -> np.ndarray:
    """Extract normalized spectral features for each frame."""
    if frames.size == 0:
        return np.zeros((0, input_size), dtype=np.float32)

    window = np.hanning(frames.shape[1]).astype(np.float32)
    spectrum = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
    log_spectrum = np.log1p(spectrum)
    resized = _resample_feature_axis(log_spectrum, input_size)
    min_vals = resized.min(axis=1, keepdims=True)
    max_vals = resized.max(axis=1, keepdims=True)
    denom = np.where((max_vals - min_vals) < 1e-8, 1.0, max_vals - min_vals)
    return ((resized - min_vals) / denom).astype(np.float32)


def rate_encode(
    normalized_features: np.ndarray,
    *,
    time_steps: int,
    threshold_scale: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Convert per-frame normalized features into spike trains."""
    if time_steps <= 0:
        raise ValueError("time_steps must be > 0")
    if normalized_features.size == 0:
        return np.zeros((0, time_steps, normalized_features.shape[1]), dtype=np.float32)

    thresholded = np.clip(normalized_features * threshold_scale, 0.0, 1.0)
    random_values = rng.random(
        (normalized_features.shape[0], time_steps, normalized_features.shape[1])
    )
    spikes = (random_values <= thresholded[:, None, :]).astype(np.float32)
    return spikes


class SpikeEncoder:
    """State-less encoder used by the signal-processing app."""

    def __init__(self, config: SpikeEncodingConfig | None = None) -> None:
        self.config = config or SpikeEncodingConfig()

    def encode_frames(self, frames: np.ndarray, *, time_steps: int, seed: int = 42) -> torch.Tensor:
        features = extract_frame_features(frames, input_size=self.config.input_size)
        rng = np.random.default_rng(seed)
        spikes = rate_encode(
            features,
            time_steps=time_steps,
            threshold_scale=self.config.threshold_scale,
            rng=rng,
        )
        return torch.from_numpy(spikes)
