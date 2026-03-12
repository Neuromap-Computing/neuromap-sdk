"""Deterministic clean/noisy pair generation with cochlear feature exports."""

from __future__ import annotations

import csv
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
from config import GammatoneFeatureConfig, ProsthesisDatasetConfig
from features import extract_gammatone_features, save_feature_tensor
from neuromap._internal.audio import load_audio_mono
from scipy.io import wavfile


def _discover_wavs(directory: Path) -> list[Path]:
    if not directory.exists():
        raise FileNotFoundError(f"Directory does not exist: {directory}")
    wavs = sorted(directory.rglob("*.wav"))
    if not wavs:
        raise ValueError(f"No .wav files found under: {directory}")
    return wavs


def _normalize_peak(audio: np.ndarray, target_peak: float) -> np.ndarray:
    if audio.size == 0:
        return audio.astype(np.float32)
    peak = float(np.max(np.abs(audio)))
    if peak <= 1e-8:
        return audio.astype(np.float32)
    return (audio * (target_peak / peak)).astype(np.float32)


def _rms(audio: np.ndarray) -> float:
    if audio.size == 0:
        return 0.0
    return float(np.sqrt(np.mean(np.square(audio), dtype=np.float64) + 1e-12))


def _fit_length(audio: np.ndarray, length: int, rng: np.random.Generator) -> np.ndarray:
    if audio.size == length:
        return audio.astype(np.float32)
    if audio.size > length:
        start = int(rng.integers(0, audio.size - length + 1))
        return audio[start : start + length].astype(np.float32)
    pad = length - audio.size
    return np.pad(audio, (0, pad), mode="constant").astype(np.float32)


def _mix_at_snr(
    clean: np.ndarray,
    noise: np.ndarray,
    *,
    snr_db: float,
    target_peak: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    clean = _normalize_peak(clean.astype(np.float32), target_peak)
    clean_rms = _rms(clean)
    noise_rms = _rms(noise)
    if clean_rms <= 1e-10 or noise_rms <= 1e-10:
        noisy = clean.copy()
        return clean, noisy, float("inf")

    target_noise_rms = clean_rms / (10 ** (snr_db / 20.0))
    scaled_noise = noise.astype(np.float32) * float(target_noise_rms / (noise_rms + 1e-12))
    noisy = clean + scaled_noise

    peak = float(max(np.max(np.abs(clean)), np.max(np.abs(noisy)), 1e-8))
    if peak > 0.999:
        gain = 0.999 / peak
        clean = clean * gain
        noisy = noisy * gain

    residual = noisy - clean
    achieved = 20.0 * np.log10((_rms(clean) + 1e-12) / (_rms(residual) + 1e-12))
    return clean.astype(np.float32), noisy.astype(np.float32), float(achieved)


def _write_wav(path: Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.int16(np.clip(audio, -1.0, 1.0) * 32767.0)
    wavfile.write(path, sample_rate, pcm)


def _build_noise_label_map(noise_metadata_csv: Path | None) -> dict[str, str]:
    if noise_metadata_csv is None or not noise_metadata_csv.exists():
        return {}
    with noise_metadata_csv.open("r", encoding="utf-8", newline="") as file_obj:
        reader = csv.DictReader(file_obj)
        return {row["slice_file_name"]: row.get("class", "unknown") for row in reader}


def _load_valid_audio_candidates(
    paths: list[Path],
    *,
    sample_rate: int,
) -> tuple[list[tuple[Path, np.ndarray]], list[Path]]:
    valid: list[tuple[Path, np.ndarray]] = []
    skipped: list[Path] = []
    for path in paths:
        try:
            audio, _ = load_audio_mono(str(path), target_sample_rate=sample_rate)
        except ValueError:
            skipped.append(path)
            continue
        valid.append((path, audio))
    return valid, skipped


def _infer_noise_type(noise_path: Path, noise_label_map: dict[str, str]) -> str:
    filename = noise_path.name
    if filename in noise_label_map:
        return noise_label_map[filename]
    if "_" in filename:
        suffix = filename.split("_", 1)[1]
        if suffix in noise_label_map:
            return noise_label_map[suffix]
    return noise_path.parent.name


def _group_key_from_clean(clean_path: Path) -> str:
    stem = clean_path.stem
    if "-" in stem:
        return stem.split("-", 1)[0]
    return stem[:5]


def _assign_splits(
    clean_paths: list[Path],
    *,
    train_ratio: float,
    val_ratio: float,
    rng: np.random.Generator,
) -> list[str]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be in (0, 1)")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be in [0, 1)")
    if train_ratio + val_ratio >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    groups = {}
    for idx, clean_path in enumerate(clean_paths):
        groups.setdefault(_group_key_from_clean(clean_path), []).append(idx)

    if len(groups) >= 3:
        group_keys = list(groups)
        rng.shuffle(group_keys)
        total = len(group_keys)
        train_cutoff = max(1, int(round(total * train_ratio)))
        val_cutoff = max(train_cutoff + 1, int(round(total * (train_ratio + val_ratio))))
        train_groups = set(group_keys[:train_cutoff])
        val_groups = set(group_keys[train_cutoff:val_cutoff])
        assignments = []
        for clean_path in clean_paths:
            key = _group_key_from_clean(clean_path)
            if key in train_groups:
                assignments.append("train")
            elif key in val_groups:
                assignments.append("val")
            else:
                assignments.append("test")
        return assignments

    indices = np.arange(len(clean_paths))
    rng.shuffle(indices)
    train_cutoff = max(1, int(round(len(indices) * train_ratio)))
    val_cutoff = max(train_cutoff + 1, int(round(len(indices) * (train_ratio + val_ratio))))
    split_map: dict[int, str] = {}
    for idx in indices[:train_cutoff]:
        split_map[int(idx)] = "train"
    for idx in indices[train_cutoff:val_cutoff]:
        split_map[int(idx)] = "val"
    for idx in indices[val_cutoff:]:
        split_map[int(idx)] = "test"
    return [split_map[i] for i in range(len(clean_paths))]


def _snr_values(cfg: ProsthesisDatasetConfig) -> np.ndarray:
    if cfg.snr_bins <= 1:
        return np.array([cfg.snr_min_db], dtype=np.float32)
    return np.linspace(cfg.snr_min_db, cfg.snr_max_db, cfg.snr_bins, dtype=np.float32)


def _build_report(
    rows: list[dict[str, Any]],
    *,
    config: ProsthesisDatasetConfig,
) -> dict[str, Any]:
    split_counts: dict[str, int] = {"train": 0, "val": 0, "test": 0}
    noise_counts: dict[str, int] = {}
    selected_snr = []
    achieved_snr = []
    duration = []
    for row in rows:
        split_counts[row["split"]] = split_counts.get(row["split"], 0) + 1
        noise = row["noise_type"]
        noise_counts[noise] = noise_counts.get(noise, 0) + 1
        selected_snr.append(float(row["selected_snr_db"]))
        achieved_snr.append(float(row["achieved_snr_db"]))
        duration.append(float(row["duration_seconds"]))

    snr_hist_counts, snr_hist_edges = np.histogram(
        np.asarray(achieved_snr, dtype=np.float32),
        bins=max(3, int(min(20, len(achieved_snr) or 3))),
    )
    duration_counts, duration_edges = np.histogram(
        np.asarray(duration, dtype=np.float32),
        bins=max(3, int(min(20, len(duration) or 3))),
    )

    return {
        "total_samples": len(rows),
        "split_counts": split_counts,
        "noise_type_counts": noise_counts,
        "selected_snr_db_mean": float(np.mean(selected_snr)) if selected_snr else 0.0,
        "achieved_snr_db_mean": float(np.mean(achieved_snr)) if achieved_snr else 0.0,
        "achieved_snr_db_std": float(np.std(achieved_snr)) if achieved_snr else 0.0,
        "duration_seconds_mean": float(np.mean(duration)) if duration else 0.0,
        "snr_histogram": {
            "edges": snr_hist_edges.tolist(),
            "counts": snr_hist_counts.tolist(),
        },
        "duration_histogram": {
            "edges": duration_edges.tolist(),
            "counts": duration_counts.tolist(),
        },
        "config": asdict(config),
    }


def generate_prosthesis_dataset(
    clean_dir: str,
    noise_dir: str,
    out_dir: str,
    *,
    config: ProsthesisDatasetConfig | dict[str, Any],
) -> dict[str, Any]:
    """Create deterministic train/val/test clean-noisy pairs and feature tensors."""
    cfg = (
        config if isinstance(config, ProsthesisDatasetConfig) else ProsthesisDatasetConfig(**config)
    )
    rng = np.random.default_rng(cfg.seed)

    clean_paths_all = _discover_wavs(Path(clean_dir))
    noise_paths_all = _discover_wavs(Path(noise_dir))
    target_samples = int(round(cfg.clip_seconds * cfg.sample_rate))
    min_duration_samples = int(round(cfg.min_duration_seconds * cfg.sample_rate))
    noise_label_map = _build_noise_label_map(
        Path(cfg.noise_metadata_csv) if cfg.noise_metadata_csv else None
    )

    clean_candidates_loaded: list[tuple[Path, np.ndarray]] = []
    for clean_path in clean_paths_all:
        clean_audio, _ = load_audio_mono(str(clean_path), target_sample_rate=cfg.sample_rate)
        if clean_audio.size >= min_duration_samples:
            clean_candidates_loaded.append((clean_path, clean_audio))

    valid_noise_loaded, skipped_noise_paths = _load_valid_audio_candidates(
        noise_paths_all, sample_rate=cfg.sample_rate
    )

    if not clean_candidates_loaded:
        raise ValueError("No clean files passed minimum-duration filtering.")
    if not valid_noise_loaded:
        raise ValueError("No readable noise files found. All candidate noise files were skipped.")
    if cfg.max_pairs is not None and cfg.max_pairs > 0:
        clean_candidates_loaded = clean_candidates_loaded[: cfg.max_pairs]

    clean_candidates = [path for path, _ in clean_candidates_loaded]
    clean_audio_map = {path: audio for path, audio in clean_candidates_loaded}
    noise_audio_map = {path: audio for path, audio in valid_noise_loaded}
    noise_paths = [path for path, _ in valid_noise_loaded]

    split_assignments = _assign_splits(
        clean_candidates,
        train_ratio=cfg.train_ratio,
        val_ratio=cfg.val_ratio,
        rng=rng,
    )
    snr_values = _snr_values(cfg)
    feature_cfg = (
        cfg.features
        if isinstance(cfg.features, GammatoneFeatureConfig)
        else GammatoneFeatureConfig(**cfg.features)
    )

    out_root = Path(out_dir)
    for split in ("train", "val", "test"):
        for key in ("clean", "noisy", "features"):
            (out_root / split / key).mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for sample_idx, (clean_path, split) in enumerate(
        zip(clean_candidates, split_assignments, strict=True)
    ):
        noise_path = noise_paths[int(rng.integers(0, len(noise_paths)))]
        snr_db = float(rng.choice(snr_values))
        clean_audio = clean_audio_map[clean_path]
        noise_audio = noise_audio_map[noise_path]

        clean_clip = _fit_length(clean_audio, target_samples, rng)
        noise_clip = _fit_length(noise_audio, target_samples, rng)
        clean_out_audio, noisy_out_audio, achieved_snr_db = _mix_at_snr(
            clean_clip,
            noise_clip,
            snr_db=snr_db,
            target_peak=cfg.target_peak,
        )

        sample_id = f"s{sample_idx:06d}"
        clean_out_path = out_root / split / "clean" / f"{sample_id}.wav"
        noisy_out_path = out_root / split / "noisy" / f"{sample_id}.wav"
        feature_path = out_root / split / "features" / f"{sample_id}.npy"

        _write_wav(clean_out_path, clean_out_audio, cfg.sample_rate)
        _write_wav(noisy_out_path, noisy_out_audio, cfg.sample_rate)
        features = extract_gammatone_features(
            noisy_out_audio,
            sample_rate=cfg.sample_rate,
            config=feature_cfg,
        )
        save_feature_tensor(feature_path, features)

        row = {
            "sample_id": sample_id,
            "split": split,
            "clean_path": str(clean_out_path),
            "noisy_path": str(noisy_out_path),
            "feature_path": str(feature_path),
            "clean_source_path": str(clean_path),
            "noise_source_path": str(noise_path),
            "selected_snr_db": round(snr_db, 4),
            "achieved_snr_db": round(achieved_snr_db, 4),
            "duration_seconds": float(target_samples / cfg.sample_rate),
            "sample_rate": cfg.sample_rate,
            "clean_rms": round(_rms(clean_out_audio), 8),
            "noisy_rms": round(_rms(noisy_out_audio), 8),
            "clean_peak": round(float(np.max(np.abs(clean_out_audio))), 8),
            "noisy_peak": round(float(np.max(np.abs(noisy_out_audio))), 8),
            "noise_type": _infer_noise_type(noise_path, noise_label_map),
            "feature_shape": list(features.shape),
        }
        rows.append(row)

    manifest_columns = [
        "sample_id",
        "split",
        "clean_path",
        "noisy_path",
        "feature_path",
        "clean_source_path",
        "noise_source_path",
        "selected_snr_db",
        "achieved_snr_db",
        "duration_seconds",
        "sample_rate",
        "clean_rms",
        "noisy_rms",
        "clean_peak",
        "noisy_peak",
        "noise_type",
        "feature_shape",
    ]
    for split in ("train", "val", "test"):
        split_rows = [row for row in rows if row["split"] == split]
        csv_path = out_root / split / "manifest.csv"
        jsonl_path = out_root / split / "manifest.jsonl"
        with csv_path.open("w", encoding="utf-8", newline="") as file_obj:
            writer = csv.DictWriter(file_obj, fieldnames=manifest_columns)
            writer.writeheader()
            for row in split_rows:
                serialized = row.copy()
                serialized["feature_shape"] = json.dumps(serialized["feature_shape"])
                writer.writerow(serialized)
        with jsonl_path.open("w", encoding="utf-8") as file_obj:
            for row in split_rows:
                file_obj.write(json.dumps(row) + "\n")

    report = _build_report(rows, config=cfg)
    report_path = out_root / "dataset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return {
        "out_dir": str(out_root),
        "num_samples": len(rows),
        "num_clean_candidates": len(clean_candidates),
        "num_noise_candidates": len(noise_paths),
        "num_noise_skipped_unreadable": len(skipped_noise_paths),
        "dataset_report": str(report_path),
    }
