"""Offline audio signal-processing application."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuromap_sim.apps.base import BaseApp
from neuromap_sim.apps.signal_processing.audio_io import frame_audio, load_audio_mono
from neuromap_sim.apps.signal_processing.encoding import SpikeEncoder, SpikeEncodingConfig
from neuromap_sim.core.quantization import quantize_weights_4bit
from neuromap_sim.models.neuromap_snn import NeuromapSNN


@dataclass(slots=True)
class SignalProcessingConfig:
    sample_rate: int = 16_000
    frame_size: int = 512
    hop_size: int = 256
    time_steps: int = 8
    input_size: int = 400
    hidden_size: int = 128
    output_size: int = 10
    apply_quantization: bool = False
    device: str = "cpu"
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SignalProcessingApp(BaseApp):
    """Plugin entrypoint for offline audio processing."""

    name = "signal-processing"

    def __init__(self) -> None:
        self.model: NeuromapSNN | None = None
        self.encoder: SpikeEncoder | None = None

    def _build_model(self, config: SignalProcessingConfig) -> NeuromapSNN:
        model = NeuromapSNN(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            output_size=config.output_size,
        ).to(config.device)
        if config.apply_quantization:
            quantize_weights_4bit(model)
        return model

    def run(self, config: dict[str, Any] | SignalProcessingConfig, **kwargs: Any) -> dict[str, Any]:
        cfg = (
            config
            if isinstance(config, SignalProcessingConfig)
            else SignalProcessingConfig(**config)
        )
        audio_path = kwargs.get("audio_path")
        if not audio_path:
            raise ValueError("audio_path is required")

        output_json = kwargs.get("output_json")
        audio, sample_rate = load_audio_mono(audio_path, target_sample_rate=cfg.sample_rate)
        frames = frame_audio(audio, frame_size=cfg.frame_size, hop_size=cfg.hop_size)

        self.encoder = SpikeEncoder(
            SpikeEncodingConfig(
                frame_size=cfg.frame_size,
                hop_size=cfg.hop_size,
                input_size=cfg.input_size,
            )
        )
        spikes = self.encoder.encode_frames(frames, time_steps=cfg.time_steps, seed=cfg.seed).to(
            cfg.device
        )
        self.model = self._build_model(cfg)

        with torch.no_grad():
            outputs = self.model(spikes)

        outputs_cpu = outputs.detach().cpu().numpy()
        result = {
            "app": self.name,
            "audio_path": str(audio_path),
            "sample_rate": int(sample_rate),
            "duration_seconds": float(audio.shape[0] / sample_rate) if sample_rate else 0.0,
            "num_frames": int(frames.shape[0]),
            "frame_size": cfg.frame_size,
            "hop_size": cfg.hop_size,
            "time_steps": cfg.time_steps,
            "model_output_shape": list(outputs.shape),
            "mean_output_activation": float(np.mean(outputs_cpu)) if outputs_cpu.size else 0.0,
            "mean_input_spike_rate": float(spikes.float().mean().item()) if spikes.numel() else 0.0,
            "config": cfg.to_dict(),
        }

        if output_json:
            output_path = Path(output_json)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
            result["output_json"] = str(output_path)

        return result
