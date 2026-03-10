"""Cochlear-inspired gammatone feature extraction."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.signal import gammatone, lfilter

from config import GammatoneFeatureConfig
from neuromap._internal.audio import frame_audio


def _erb_space(low_hz: float, high_hz: float, n_filters: int) -> np.ndarray:
    if n_filters <= 0:
        raise ValueError("n_filters must be > 0")
    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("Invalid frequency range for gammatone filterbank.")
    low_erb = 21.4 * np.log10(4.37e-3 * low_hz + 1.0)
    high_erb = 21.4 * np.log10(4.37e-3 * high_hz + 1.0)
    erb_points = np.linspace(low_erb, high_erb, n_filters)
    return (10 ** (erb_points / 21.4) - 1.0) / 4.37e-3


def _compress(values: np.ndarray, cfg: GammatoneFeatureConfig) -> np.ndarray:
    if cfg.compression == "log1p":
        return np.log1p(values)
    if cfg.compression == "power":
        return np.power(values, cfg.power).astype(np.float32)
    raise ValueError(f"Unsupported compression mode: {cfg.compression}")


def extract_gammatone_features(
    audio: np.ndarray,
    *,
    sample_rate: int,
    config: GammatoneFeatureConfig,
) -> np.ndarray:
    """Convert mono audio into frame-level cochlear features."""
    if audio.ndim != 1:
        raise ValueError("Expected mono 1D audio array.")
    if sample_rate != config.sample_rate:
        raise ValueError(
            f"Sample rate mismatch. Got {sample_rate}, expected {config.sample_rate}. "
            "Resample audio before feature extraction."
        )
    if audio.size == 0:
        return np.zeros((0, config.n_filters), dtype=np.float32)

    center_freqs = _erb_space(config.low_freq_hz, config.high_freq_hz, config.n_filters)
    channel_features: list[np.ndarray] = []
    for center_hz in center_freqs:
        b, a = gammatone(center_hz, "iir", fs=sample_rate)
        band_signal = lfilter(b, a, audio).astype(np.float32)
        envelope = np.abs(band_signal)
        compressed = _compress(envelope, config)
        frames = frame_audio(compressed, frame_size=config.frame_size, hop_size=config.hop_size)
        channel_features.append(frames.mean(axis=1, dtype=np.float32))

    features = np.stack(channel_features, axis=1).astype(np.float32)
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    features = (features - mean) / (std + 1e-6)
    return features.astype(np.float32)


def save_feature_tensor(feature_path: Path, features: np.ndarray) -> None:
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(feature_path, features.astype(np.float32), allow_pickle=False)
