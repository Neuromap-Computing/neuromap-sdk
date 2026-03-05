"""Prosthesis app plugin entrypoint."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from neuromap_sim.apps.base import BaseApp
from neuromap_sim.apps.prosthesis.config import (
    GammatoneFeatureConfig,
    ProsthesisBootstrapConfig,
    ProsthesisDatasetConfig,
    ProsthesisExportConfig,
    ProsthesisTrainConfig,
)
from neuromap_sim.apps.prosthesis.data_bootstrap import bootstrap_prosthesis_raw_data
from neuromap_sim.apps.prosthesis.data_generation import generate_prosthesis_dataset
from neuromap_sim.apps.prosthesis.dataset import (
    ProsthesisDenoisingDataset,
    collate_batch,
    read_manifest_rows,
)


class ProsthesisApp(BaseApp):
    """App facade for dataset bootstrap, generation, training, and export flows."""

    name = "prosthesis"

    def run(self, config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action")

        if action == "bootstrap":
            return self._run_bootstrap(config)

        if action == "generate":
            return self._run_generate(config, **kwargs)

        if action == "train":
            return self._run_train(config, **kwargs)

        if action == "export":
            return self._run_export(config, **kwargs)

        raise ValueError("Unknown action. Expected one of: bootstrap, generate, train, export.")

    # -- bootstrap ----------------------------------------------------------------

    def _run_bootstrap(self, config: Any) -> dict[str, Any]:
        cfg = (
            config
            if isinstance(config, ProsthesisBootstrapConfig)
            else ProsthesisBootstrapConfig(**config)
        )
        result = bootstrap_prosthesis_raw_data(cfg)
        return {"app": self.name, "action": "bootstrap", "config": cfg.to_dict(), **result}

    # -- generate -----------------------------------------------------------------

    def _run_generate(self, config: Any, **kwargs: Any) -> dict[str, Any]:
        cfg = (
            config
            if isinstance(config, ProsthesisDatasetConfig)
            else ProsthesisDatasetConfig(**config)
        )
        clean_dir = kwargs.get("clean_dir")
        noise_dir = kwargs.get("noise_dir")
        out_dir = kwargs.get("out_dir")
        if not clean_dir or not noise_dir or not out_dir:
            raise ValueError("clean_dir, noise_dir and out_dir are required for generate action.")
        result = generate_prosthesis_dataset(
            clean_dir=clean_dir,
            noise_dir=noise_dir,
            out_dir=out_dir,
            config=cfg,
        )
        return {"app": self.name, "action": "generate", "config": cfg.to_dict(), **result}

    # -- train (SDK-powered) ------------------------------------------------------

    def _run_train(self, config: Any, **kwargs: Any) -> dict[str, Any]:
        from neuromap_sim.sdk import Network, Trainer, chips

        cfg = (
            config
            if isinstance(config, ProsthesisTrainConfig)
            else ProsthesisTrainConfig(**config)
        )

        data_dir = Path(kwargs.get("data_dir", "data/prosthesis/generated"))
        output_dir = Path(kwargs.get("output_dir", "data/prosthesis/training_runs"))
        output_dir.mkdir(parents=True, exist_ok=True)

        # Seed
        torch.manual_seed(cfg.seed)

        # Load manifests
        train_rows = read_manifest_rows(data_dir / "train" / "manifest.csv")
        val_rows = read_manifest_rows(data_dir / "val" / "manifest.csv")
        if not train_rows:
            raise ValueError(f"No training rows found at: {data_dir / 'train' / 'manifest.csv'}")

        if cfg.max_train_rows > 0:
            train_rows = train_rows[: cfg.max_train_rows]
        if cfg.max_val_rows > 0:
            val_rows = val_rows[: cfg.max_val_rows]

        # Auto-detect input size from the first feature file
        input_size = int(np.load(train_rows[0]["feature_path"], allow_pickle=False).shape[1])
        feature_config = GammatoneFeatureConfig(
            sample_rate=cfg.feature_sample_rate,
            n_filters=input_size,
            low_freq_hz=cfg.features.low_freq_hz,
            high_freq_hz=cfg.features.high_freq_hz,
            frame_size=cfg.features.frame_size,
            hop_size=cfg.features.hop_size,
            compression=cfg.features.compression,
            power=cfg.features.power,
        )

        # Build network from chip spec or custom topology
        if cfg.chip_name:
            chip = chips.get(cfg.chip_name)
            net = Network(chip, use_decoder=cfg.use_decoder)
        else:
            net = Network.from_topology(
                layers=[input_size, cfg.hidden_size, input_size],
                name="prosthesis-auto",
                use_decoder=cfg.use_decoder,
            )

        # Resolve device
        device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Build datasets + loaders
        train_ds = ProsthesisDenoisingDataset(
            train_rows,
            sample_rate=cfg.feature_sample_rate,
            feature_config=feature_config,
            target_cache_size=cfg.target_cache_size,
        )
        val_ds = ProsthesisDenoisingDataset(
            val_rows,
            sample_rate=cfg.feature_sample_rate,
            feature_config=feature_config,
            target_cache_size=cfg.target_cache_size,
        ) if val_rows else None

        train_loader = DataLoader(
            train_ds,
            batch_size=cfg.batch_size,
            shuffle=True,
            num_workers=cfg.num_workers,
            collate_fn=collate_batch,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_batch,
        ) if val_ds is not None else None

        # Build composite loss matching the original training script
        def composite_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
            alpha = cfg.loss_alpha
            mse = (preds - targets) ** 2
            l1 = torch.abs(preds - targets)
            return (alpha * mse + (1.0 - alpha) * l1).mean()

        # Train with SDK Trainer
        trainer = Trainer(
            net,
            lr=cfg.learning_rate,
            epochs=cfg.epochs,
            device=device,
            weight_decay=cfg.weight_decay,
            scheduler="plateau",
            scheduler_factor=cfg.scheduler_factor,
            scheduler_patience=cfg.scheduler_patience,
            min_lr=cfg.min_learning_rate,
            lr_warmup_epochs=cfg.lr_warmup_epochs,
            early_stop_patience=cfg.early_stop_patience,
            early_stop_min_delta=cfg.early_stop_min_delta,
            grad_clip_norm=cfg.grad_clip_norm,
            loss_fn=composite_loss,
            use_sequence_forward=True,
        )

        history = trainer.fit(train_loader, val_loader)

        # Save checkpoint
        checkpoint_path = output_dir / "best_prosthesis_snn_cls.pt"
        net.save(checkpoint_path)

        # Save training history
        history_path = output_dir / "training_history.json"
        history_path.write_text(
            json.dumps({"history": history.to_list()}, indent=2),
            encoding="utf-8",
        )

        return {
            "app": self.name,
            "action": "train",
            "config": cfg.to_dict(),
            "checkpoint": str(checkpoint_path),
            "history_path": str(history_path),
            "epochs_trained": len(history.epochs),
            "final_train_loss": history.epochs[-1]["train_loss"] if history.epochs else None,
            "final_val_loss": history.epochs[-1].get("val_loss") if history.epochs else None,
            "network_summary": net.summary(),
        }

    # -- export (SDK-powered) -----------------------------------------------------

    def _run_export(self, config: Any, **kwargs: Any) -> dict[str, Any]:
        from neuromap_sim.sdk import Exporter, Network

        cfg = (
            config
            if isinstance(config, ProsthesisExportConfig)
            else ProsthesisExportConfig(**config)
        )

        checkpoint = kwargs.get("checkpoint")
        if not checkpoint:
            raise ValueError("checkpoint is required for export action.")
        output_path = kwargs.get("output_path", "model_export.nmap")

        # Load network from checkpoint
        device = kwargs.get("device", "cpu")
        net = Network.load(checkpoint, device=device)

        # Quantize and export
        exporter = Exporter(net)
        exporter.quantize(bits=cfg.bits)
        exporter.save(output_path)

        weight_map = exporter.to_weight_map()
        weight_shapes = {k: list(v.shape) for k, v in weight_map.items()}

        return {
            "app": self.name,
            "action": "export",
            "config": cfg.to_dict(),
            "checkpoint": str(checkpoint),
            "output_path": str(output_path),
            "weight_bits": cfg.bits,
            "weight_shapes": weight_shapes,
        }
