"""Bootstrap raw clean/noise datasets with reproducible staging layout."""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any, Callable

from config import ProsthesisBootstrapConfig


def _copy_wavs(
    source_files: list[Path],
    destination_dir: Path,
    *,
    filename_transform: Callable[[Path], str] | None = None,
    limit: int | None = None,
) -> int:
    destination_dir.mkdir(parents=True, exist_ok=True)
    copied = 0
    for file_path in sorted(source_files):
        if limit is not None and copied >= limit:
            break
        target_name = (
            file_path.name if filename_transform is None else filename_transform(file_path)
        )
        target_path = destination_dir / target_name
        if not target_path.exists():
            shutil.copy2(file_path, target_path)
        copied += 1
    return copied


def _bootstrap_ljspeech(raw_root: Path, clean_out: Path, limit: int | None) -> dict[str, Any]:
    try:
        from torchaudio.datasets import LJSPEECH
    except Exception as exc:
        raise RuntimeError(
            "torchaudio is required for LJSpeech bootstrap. Install dependency and retry."
        ) from exc

    ljspeech_cache = raw_root / "ljspeech"
    ljspeech_cache.mkdir(parents=True, exist_ok=True)
    LJSPEECH(root=str(ljspeech_cache), download=True)
    wavs_dir = ljspeech_cache / "LJSpeech-1.1" / "wavs"
    clean_files = list(wavs_dir.glob("*.wav"))
    copied = _copy_wavs(clean_files, clean_out, limit=limit)
    return {"clean_files_found": len(clean_files), "clean_files_staged": copied}


def _find_urbansound8k_root(urban_cache_root: Path) -> Path:
    if (urban_cache_root / "audio").exists() and (urban_cache_root / "metadata").exists():
        return urban_cache_root

    candidate = urban_cache_root / "urbansound8k"
    if candidate.exists():
        return candidate
    candidate_upper = urban_cache_root / "UrbanSound8K"
    if candidate_upper.exists():
        return candidate_upper

    if urban_cache_root.exists():
        for child in urban_cache_root.iterdir():
            if child.is_dir() and (
                child.name.lower() == "urbansound8k"
                or ((child / "audio").exists() and (child / "metadata").exists())
            ):
                return child
    raise FileNotFoundError(
        f"Unable to locate UrbanSound8K root under '{urban_cache_root}'. "
        "Expected a folder named 'urbansound8k'."
    )


def _bootstrap_urbansound8k(
    raw_root: Path,
    noise_out: Path,
    metadata_export_path: Path,
    limit: int | None,
) -> dict[str, Any]:
    try:
        import soundata
    except Exception as exc:
        raise RuntimeError(
            "soundata is required for UrbanSound8K bootstrap. Install dependency and retry."
        ) from exc

    urbansound_cache = raw_root / "urbansound8k"
    urbansound_cache.mkdir(parents=True, exist_ok=True)
    dataset = soundata.initialize("urbansound8k", data_home=str(urbansound_cache))
    dataset.download()
    dataset.validate()

    dataset_root = _find_urbansound8k_root(urbansound_cache)
    audio_root = dataset_root / "audio"
    noise_files = list(audio_root.glob("fold*/*.wav"))
    copied = _copy_wavs(
        noise_files,
        noise_out,
        filename_transform=lambda p: f"{p.parent.name}_{p.name}",
        limit=limit,
    )

    metadata_path = dataset_root / "metadata" / "UrbanSound8K.csv"
    metadata_export_path.parent.mkdir(parents=True, exist_ok=True)
    rows_written = 0
    with metadata_path.open("r", encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        with metadata_export_path.open("w", encoding="utf-8", newline="") as output_file:
            fieldnames = ["slice_file_name", "fold", "class", "salience"]
            writer = csv.DictWriter(output_file, fieldnames=fieldnames)
            writer.writeheader()
            for row in reader:
                writer.writerow({name: row[name] for name in fieldnames})
                rows_written += 1

    return {
        "noise_files_found": len(noise_files),
        "noise_files_staged": copied,
        "noise_metadata_csv": str(metadata_export_path),
        "noise_metadata_rows": rows_written,
    }


def bootstrap_prosthesis_raw_data(
    config: ProsthesisBootstrapConfig | dict[str, Any],
) -> dict[str, Any]:
    """Download and stage clean/noise WAV sources for prosthesis generation."""
    cfg = (
        config
        if isinstance(config, ProsthesisBootstrapConfig)
        else ProsthesisBootstrapConfig(**config)
    )

    raw_root = Path(cfg.raw_root)
    clean_out = raw_root / "clean_source"
    noise_out = raw_root / "noise_source"
    metadata_export = raw_root / "urbansound8k_metadata_min.csv"

    raw_root.mkdir(parents=True, exist_ok=True)
    bootstrap_clean = _bootstrap_ljspeech(raw_root, clean_out, cfg.clean_limit)
    bootstrap_noise = _bootstrap_urbansound8k(raw_root, noise_out, metadata_export, cfg.noise_limit)

    return {
        "raw_root": str(raw_root),
        "clean_dir": str(clean_out),
        "noise_dir": str(noise_out),
        "ljspeech_cache": str(raw_root / "ljspeech"),
        "urbansound8k_cache": str(raw_root / "urbansound8k"),
        **bootstrap_clean,
        **bootstrap_noise,
    }
