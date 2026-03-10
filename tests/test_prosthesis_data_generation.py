"""Tests for the prosthesis data generation pipeline."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

# Add examples/prosthesis to path so we can import the example modules
_EXAMPLES_DIR = Path(__file__).resolve().parent.parent / "examples" / "prosthesis"
if str(_EXAMPLES_DIR) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES_DIR))

from config import GammatoneFeatureConfig, ProsthesisDatasetConfig  # noqa: E402
from data_generation import generate_prosthesis_dataset  # noqa: E402
import data_generation  # noqa: E402
import data_bootstrap  # noqa: E402
from data_bootstrap import bootstrap_prosthesis_raw_data  # noqa: E402
from config import ProsthesisBootstrapConfig  # noqa: E402


def _read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as file_obj:
        return list(csv.DictReader(file_obj))


def _collect_selected_snrs(out_dir: Path) -> list[float]:
    values = []
    for split in ("train", "val", "test"):
        manifest = _read_manifest(out_dir / split / "manifest.csv")
        for row in manifest:
            values.append(float(row["selected_snr_db"]))
    return values


def test_generate_prosthesis_dataset_deterministic(synthetic_audio_corpus: dict[str, Path]) -> None:
    clean_dir = synthetic_audio_corpus["clean_dir"]
    noise_dir = synthetic_audio_corpus["noise_dir"]
    metadata_csv = synthetic_audio_corpus["noise_metadata_csv"]

    out_a = clean_dir.parent / "generated_a"
    out_b = clean_dir.parent / "generated_b"
    cfg = ProsthesisDatasetConfig(
        sample_rate=16_000,
        clip_seconds=0.75,
        snr_min_db=3.0,
        snr_max_db=12.0,
        snr_bins=4,
        train_ratio=0.5,
        val_ratio=0.25,
        seed=99,
        noise_metadata_csv=str(metadata_csv),
        features=GammatoneFeatureConfig(sample_rate=16_000, n_filters=12, frame_size=128, hop_size=64),
    )

    result_a = generate_prosthesis_dataset(str(clean_dir), str(noise_dir), str(out_a), config=cfg)
    result_b = generate_prosthesis_dataset(str(clean_dir), str(noise_dir), str(out_b), config=cfg)

    assert result_a["num_samples"] == result_b["num_samples"] > 0
    assert _collect_selected_snrs(out_a) == _collect_selected_snrs(out_b)


def test_generate_prosthesis_dataset_outputs_and_snr(synthetic_audio_corpus: dict[str, Path]) -> None:
    clean_dir = synthetic_audio_corpus["clean_dir"]
    noise_dir = synthetic_audio_corpus["noise_dir"]
    out_dir = clean_dir.parent / "generated"
    cfg = ProsthesisDatasetConfig(
        sample_rate=16_000,
        clip_seconds=0.6,
        snr_min_db=0.0,
        snr_max_db=15.0,
        snr_bins=6,
        train_ratio=0.5,
        val_ratio=0.25,
        seed=7,
        features=GammatoneFeatureConfig(sample_rate=16_000, n_filters=10, frame_size=128, hop_size=64),
    )

    generate_prosthesis_dataset(str(clean_dir), str(noise_dir), str(out_dir), config=cfg)

    all_rows: list[dict[str, str]] = []
    for split in ("train", "val", "test"):
        manifest_csv = out_dir / split / "manifest.csv"
        manifest_jsonl = out_dir / split / "manifest.jsonl"
        assert manifest_csv.exists()
        assert manifest_jsonl.exists()
        all_rows.extend(_read_manifest(manifest_csv))

    assert all_rows
    for row in all_rows:
        clean_path = Path(row["clean_path"])
        noisy_path = Path(row["noisy_path"])
        feature_path = Path(row["feature_path"])
        assert clean_path.exists()
        assert noisy_path.exists()
        assert feature_path.exists()

        selected = float(row["selected_snr_db"])
        achieved = float(row["achieved_snr_db"])
        assert abs(selected - achieved) < 1.6

        shape = json.loads(row["feature_shape"])
        features = np.load(feature_path, allow_pickle=False)
        assert list(features.shape) == shape
        assert features.ndim == 2
        assert features.shape[1] == 10

    report = json.loads((out_dir / "dataset_report.json").read_text(encoding="utf-8"))
    assert report["total_samples"] == len(all_rows)
    assert set(report["split_counts"]) == {"train", "val", "test"}


def test_bootstrap_orchestration_with_monkeypatch(tmp_path: Path, monkeypatch) -> None:
    def fake_clean(_raw_root: Path, clean_out: Path, _limit: int | None) -> dict[str, int]:
        clean_out.mkdir(parents=True, exist_ok=True)
        (clean_out / "dummy_clean.wav").write_bytes(b"")
        return {"clean_files_found": 1, "clean_files_staged": 1}

    def fake_noise(
        _raw_root: Path,
        noise_out: Path,
        metadata_export_path: Path,
        _limit: int | None,
    ) -> dict[str, int | str]:
        noise_out.mkdir(parents=True, exist_ok=True)
        (noise_out / "dummy_noise.wav").write_bytes(b"")
        metadata_export_path.write_text(
            "slice_file_name,fold,class,salience\nx.wav,1,siren,1\n",
            encoding="utf-8",
        )
        return {
            "noise_files_found": 1,
            "noise_files_staged": 1,
            "noise_metadata_csv": str(metadata_export_path),
            "noise_metadata_rows": 1,
        }

    monkeypatch.setattr(data_bootstrap, "_bootstrap_ljspeech", fake_clean)
    monkeypatch.setattr(data_bootstrap, "_bootstrap_urbansound8k", fake_noise)

    result = bootstrap_prosthesis_raw_data(
        ProsthesisBootstrapConfig(raw_root=str(tmp_path / "raw"), clean_limit=1, noise_limit=1)
    )
    assert result["clean_files_staged"] == 1
    assert result["noise_files_staged"] == 1
    assert Path(result["clean_dir"]).exists()
    assert Path(result["noise_dir"]).exists()


def test_generate_skips_unreadable_noise(synthetic_audio_corpus: dict[str, Path], monkeypatch) -> None:
    clean_dir = synthetic_audio_corpus["clean_dir"]
    noise_dir = synthetic_audio_corpus["noise_dir"]
    out_dir = clean_dir.parent / "generated_skip_bad_noise"
    bad_noise = noise_dir / "fold1_bad_adpcm.wav"
    bad_noise.write_bytes(b"not-a-valid-wav")

    original_loader = data_generation.load_audio_mono

    def patched_loader(audio_path: str, *, target_sample_rate: int | None = None):
        if audio_path.endswith("fold1_bad_adpcm.wav"):
            raise ValueError("Unknown wave file format: ADPCM")
        return original_loader(audio_path, target_sample_rate=target_sample_rate)

    monkeypatch.setattr(data_generation, "load_audio_mono", patched_loader)
    cfg = ProsthesisDatasetConfig(
        sample_rate=16_000,
        clip_seconds=0.6,
        snr_min_db=0.0,
        snr_max_db=10.0,
        snr_bins=3,
        train_ratio=0.5,
        val_ratio=0.25,
        seed=11,
        features=GammatoneFeatureConfig(sample_rate=16_000, n_filters=8, frame_size=128, hop_size=64),
    )

    result = generate_prosthesis_dataset(str(clean_dir), str(noise_dir), str(out_dir), config=cfg)
    assert result["num_samples"] > 0
    assert result["num_noise_skipped_unreadable"] >= 1
