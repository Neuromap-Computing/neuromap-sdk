"""Shared test fixtures for the neuromap test suite."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.io import wavfile


@pytest.fixture
def synthetic_audio_corpus(tmp_path: Path) -> dict[str, Path]:
    """Create small clean/noise WAV fixtures for dataset generation tests."""
    sample_rate = 16_000
    duration_s = 1.2
    timeline = np.linspace(0, duration_s, int(sample_rate * duration_s), endpoint=False)

    clean_dir = tmp_path / "clean_source"
    noise_dir = tmp_path / "noise_source"
    clean_dir.mkdir(parents=True, exist_ok=True)
    noise_dir.mkdir(parents=True, exist_ok=True)

    for idx, freq in enumerate((220, 330, 440, 550)):
        clean = 0.4 * np.sin(2 * np.pi * freq * timeline)
        wavfile.write(
            clean_dir / f"LJ001-{idx:04d}.wav",
            sample_rate,
            np.int16(np.clip(clean, -1.0, 1.0) * 32767),
        )

    rng = np.random.default_rng(1234)
    for idx in range(6):
        noise = 0.18 * rng.standard_normal(timeline.shape[0], dtype=np.float32)
        wavfile.write(
            noise_dir / f"fold{(idx % 3) + 1}_noise_{idx:02d}.wav",
            sample_rate,
            np.int16(np.clip(noise, -1.0, 1.0) * 32767),
        )

    metadata_csv = tmp_path / "urbansound8k_metadata_min.csv"
    metadata_csv.write_text(
        "slice_file_name,fold,class,salience\n"
        "noise_00.wav,1,air_conditioner,1\n"
        "noise_01.wav,2,siren,2\n",
        encoding="utf-8",
    )
    return {
        "clean_dir": clean_dir,
        "noise_dir": noise_dir,
        "noise_metadata_csv": metadata_csv,
        "sample_rate": sample_rate,
    }
