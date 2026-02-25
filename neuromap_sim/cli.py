"""Command line entrypoint for Neuromap simulator apps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from neuromap_sim.apps.prosthesis import (
    GammatoneFeatureConfig,
    ProsthesisApp,
    ProsthesisBootstrapConfig,
    ProsthesisDatasetConfig,
)
from neuromap_sim.apps.registry import AppRegistry
from neuromap_sim.apps.signal_processing import SignalProcessingApp, SignalProcessingConfig
from neuromap_sim.core.quantization import quantize_weights_4bit
from neuromap_sim.models.neuromap_snn import NeuromapSNN


def _resolve_output_json_path(path_arg: str, *, command_name: str) -> Path:
    output_path = Path(path_arg)
    if output_path.exists() and output_path.is_dir():
        return output_path / f"{command_name}_summary.json"
    if path_arg.endswith("/") or path_arg.endswith("\\"):
        output_path.mkdir(parents=True, exist_ok=True)
        return output_path / f"{command_name}_summary.json"
    return output_path


def build_registry() -> AppRegistry:
    registry = AppRegistry()
    registry.register(SignalProcessingApp())
    registry.register(ProsthesisApp())
    return registry


def run_demo() -> dict[str, Any]:
    torch.manual_seed(42)
    model = NeuromapSNN()
    x = torch.rand(2, 8, 400)
    y = model(x)
    quantize_weights_4bit(model)
    y_q = model(x)
    return {
        "input_shape": list(x.shape),
        "output_shape": list(y.shape),
        "quantized_output_shape": list(y_q.shape),
        "dtype": str(y.dtype),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Neuromap SNN toolkit")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-apps", help="List registered app plugins")
    subparsers.add_parser("demo", help="Run a quick random-tensor SNN demo")

    process_audio = subparsers.add_parser("process-audio", help="Run offline audio processing")
    process_audio.add_argument("--audio-path", required=True, help="Path to input WAV file")
    process_audio.add_argument("--output-json", help="Optional output JSON summary path")
    process_audio.add_argument("--sample-rate", type=int, default=16_000)
    process_audio.add_argument("--frame-size", type=int, default=512)
    process_audio.add_argument("--hop-size", type=int, default=256)
    process_audio.add_argument("--time-steps", type=int, default=8)
    process_audio.add_argument("--input-size", type=int, default=400)
    process_audio.add_argument("--hidden-size", type=int, default=128)
    process_audio.add_argument("--output-size", type=int, default=10)
    process_audio.add_argument("--device", default="cpu")
    process_audio.add_argument("--seed", type=int, default=42)
    process_audio.add_argument("--apply-quantization", action="store_true")

    bootstrap_data = subparsers.add_parser(
        "bootstrap-prosthesis-data",
        help="Download and stage clean/noise raw datasets for prosthesis generation",
    )
    bootstrap_data.add_argument("--raw-root", default="data/prosthesis/raw")
    bootstrap_data.add_argument("--clean-limit", type=int)
    bootstrap_data.add_argument("--noise-limit", type=int)
    bootstrap_data.add_argument("--output-json", help="Optional output JSON summary path")

    generate_data = subparsers.add_parser(
        "generate-prosthesis-dataset",
        help="Generate deterministic clean/noisy pairs with gammatone features",
    )
    generate_data.add_argument("--clean-dir", required=True)
    generate_data.add_argument("--noise-dir", required=True)
    generate_data.add_argument("--out-dir", required=True)
    generate_data.add_argument("--noise-metadata-csv")
    generate_data.add_argument("--sample-rate", type=int, default=16_000)
    generate_data.add_argument("--clip-seconds", type=float, default=4.0)
    generate_data.add_argument("--min-duration-seconds", type=float, default=0.3)
    generate_data.add_argument("--snr-min", type=float, default=0.0)
    generate_data.add_argument("--snr-max", type=float, default=20.0)
    generate_data.add_argument("--snr-bins", type=int, default=5)
    generate_data.add_argument("--train-ratio", type=float, default=0.8)
    generate_data.add_argument("--val-ratio", type=float, default=0.1)
    generate_data.add_argument("--target-peak", type=float, default=0.9)
    generate_data.add_argument("--max-pairs", type=int)
    generate_data.add_argument("--seed", type=int, default=42)
    generate_data.add_argument("--gt-n-filters", type=int, default=32)
    generate_data.add_argument("--gt-low-freq-hz", type=float, default=80.0)
    generate_data.add_argument("--gt-high-freq-hz", type=float, default=7_600.0)
    generate_data.add_argument("--gt-frame-size", type=int, default=256)
    generate_data.add_argument("--gt-hop-size", type=int, default=128)
    generate_data.add_argument("--gt-compression", choices=("log1p", "power"), default="log1p")
    generate_data.add_argument("--gt-power", type=float, default=0.3)
    generate_data.add_argument("--output-json", help="Optional output JSON summary path")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    registry = build_registry()

    if args.command == "list-apps":
        print("\n".join(registry.list_apps()))
        return

    if args.command == "demo":
        print(json.dumps(run_demo(), indent=2))
        return

    if args.command == "process-audio":
        app = registry.get("signal-processing")
        config = SignalProcessingConfig(
            sample_rate=args.sample_rate,
            frame_size=args.frame_size,
            hop_size=args.hop_size,
            time_steps=args.time_steps,
            input_size=args.input_size,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            apply_quantization=args.apply_quantization,
            device=args.device,
            seed=args.seed,
        )
        result = app.run(config, audio_path=args.audio_path, output_json=args.output_json)
        print(json.dumps(result, indent=2))
        return

    if args.command == "bootstrap-prosthesis-data":
        app = registry.get("prosthesis")
        config = ProsthesisBootstrapConfig(
            raw_root=args.raw_root,
            clean_limit=args.clean_limit,
            noise_limit=args.noise_limit,
        )
        result = app.run(config, action="bootstrap")
        if args.output_json:
            output_path = _resolve_output_json_path(
                args.output_json, command_name="bootstrap_prosthesis_data"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(result, indent=2))
            result["output_json"] = str(output_path)
        print(json.dumps(result, indent=2))
        return

    if args.command == "generate-prosthesis-dataset":
        app = registry.get("prosthesis")
        config = ProsthesisDatasetConfig(
            sample_rate=args.sample_rate,
            clip_seconds=args.clip_seconds,
            min_duration_seconds=args.min_duration_seconds,
            snr_min_db=args.snr_min,
            snr_max_db=args.snr_max,
            snr_bins=args.snr_bins,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            target_peak=args.target_peak,
            seed=args.seed,
            max_pairs=args.max_pairs,
            noise_metadata_csv=args.noise_metadata_csv,
            features=GammatoneFeatureConfig(
                sample_rate=args.sample_rate,
                n_filters=args.gt_n_filters,
                low_freq_hz=args.gt_low_freq_hz,
                high_freq_hz=args.gt_high_freq_hz,
                frame_size=args.gt_frame_size,
                hop_size=args.gt_hop_size,
                compression=args.gt_compression,
                power=args.gt_power,
            ),
        )
        result = app.run(
            config,
            action="generate",
            clean_dir=args.clean_dir,
            noise_dir=args.noise_dir,
            out_dir=args.out_dir,
        )
        if args.output_json:
            output_path = _resolve_output_json_path(
                args.output_json, command_name="generate_prosthesis_dataset"
            )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("w", encoding="utf-8") as file_obj:
                file_obj.write(json.dumps(result, indent=2))
            result["output_json"] = str(output_path)
        print(json.dumps(result, indent=2))
        return

    raise RuntimeError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    main()
