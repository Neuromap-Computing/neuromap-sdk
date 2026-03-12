#!/usr/bin/env python3
"""Prosthesis cochlear-implant denoising — standalone example.

Usage::

    # 1. Bootstrap raw datasets (downloads LJSpeech + UrbanSound8K)
    python run.py bootstrap

    # 2. Generate train/val/test pairs
    python run.py generate --clean-dir data/raw/clean_source --noise-dir data/raw/noise_source --out-dir data/generated

    # 3. Train the SNN
    python run.py train --data-dir data/generated --output-dir runs/

    # 4. Export to .nmap
    python run.py export --checkpoint runs/best_prosthesis_snn_cls.pt --output model.nmap
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from config import (
    GammatoneFeatureConfig,
    ProsthesisBootstrapConfig,
    ProsthesisDatasetConfig,
    ProsthesisExportConfig,
    ProsthesisTrainConfig,
)
from data_bootstrap import bootstrap_prosthesis_raw_data
from data_generation import generate_prosthesis_dataset
from dataset import ProsthesisDenoisingDataset, collate_batch, read_manifest_rows
from neuromap import Exporter, Network, Trainer, chips
from torch.utils.data import DataLoader


def cmd_bootstrap(args: argparse.Namespace) -> None:
    """Download and stage raw audio datasets."""
    cfg = ProsthesisBootstrapConfig(
        raw_root=args.raw_root,
        clean_limit=args.clean_limit,
        noise_limit=args.noise_limit,
    )
    result = bootstrap_prosthesis_raw_data(cfg)
    print(json.dumps(result, indent=2))


def cmd_generate(args: argparse.Namespace) -> None:
    """Generate clean/noisy pairs and cochlear features."""
    cfg = ProsthesisDatasetConfig(
        sample_rate=args.sample_rate,
        clip_seconds=args.clip_seconds,
        seed=args.seed,
    )
    result = generate_prosthesis_dataset(
        clean_dir=args.clean_dir,
        noise_dir=args.noise_dir,
        out_dir=args.out_dir,
        config=cfg,
    )
    print(json.dumps(result, indent=2))


def cmd_train(args: argparse.Namespace) -> None:
    """Train the prosthesis SNN."""
    cfg = ProsthesisTrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
    )

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(cfg.seed)

    train_rows = read_manifest_rows(data_dir / "train" / "manifest.csv")
    val_rows = read_manifest_rows(data_dir / "val" / "manifest.csv")
    if not train_rows:
        raise ValueError(f"No training rows found at: {data_dir / 'train' / 'manifest.csv'}")

    input_size = int(np.load(train_rows[0]["feature_path"], allow_pickle=False).shape[1])
    feature_config = GammatoneFeatureConfig(
        sample_rate=cfg.feature_sample_rate,
        n_filters=input_size,
    )

    if cfg.chip_name:
        chip = chips.get(cfg.chip_name)
        net = Network(chip, use_decoder=cfg.use_decoder)
    else:
        net = Network.from_topology(
            layers=[input_size, cfg.hidden_size, input_size],
            name="prosthesis-auto",
            use_decoder=cfg.use_decoder,
        )

    device = cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = ProsthesisDenoisingDataset(
        train_rows,
        sample_rate=cfg.feature_sample_rate,
        feature_config=feature_config,
        target_cache_size=cfg.target_cache_size,
    )
    val_ds = (
        ProsthesisDenoisingDataset(
            val_rows,
            sample_rate=cfg.feature_sample_rate,
            feature_config=feature_config,
            target_cache_size=cfg.target_cache_size,
        )
        if val_rows
        else None
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        collate_fn=collate_batch,
    )
    val_loader = (
        DataLoader(
            val_ds,
            batch_size=cfg.batch_size,
            shuffle=False,
            num_workers=cfg.num_workers,
            collate_fn=collate_batch,
        )
        if val_ds is not None
        else None
    )

    def composite_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        alpha = cfg.loss_alpha
        mse = (preds - targets) ** 2
        l1 = torch.abs(preds - targets)
        return (alpha * mse + (1.0 - alpha) * l1).mean()

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

    checkpoint_path = output_dir / "best_prosthesis_snn_cls.pt"
    net.save(checkpoint_path)

    history_path = output_dir / "training_history.json"
    history_path.write_text(
        json.dumps({"history": history.to_list()}, indent=2),
        encoding="utf-8",
    )

    print(f"Checkpoint saved to {checkpoint_path}")
    print(f"History saved to {history_path}")
    print(net.summary())


def cmd_export(args: argparse.Namespace) -> None:
    """Export a trained model to .nmap format."""
    cfg = ProsthesisExportConfig(bits=args.bits)
    net = Network.load(args.checkpoint, device=args.device)
    exporter = Exporter(net)
    exporter.quantize(bits=cfg.bits)
    exporter.save(args.output)

    weight_map = exporter.to_weight_map()
    for name, arr in weight_map.items():
        print(f"  {name}: {arr.shape}")
    print(f"Exported to {args.output}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prosthesis cochlear-implant denoising pipeline",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # bootstrap
    p_boot = sub.add_parser("bootstrap", help="Download raw audio datasets")
    p_boot.add_argument("--raw-root", default="data/prosthesis/raw")
    p_boot.add_argument("--clean-limit", type=int, default=None)
    p_boot.add_argument("--noise-limit", type=int, default=None)

    # generate
    p_gen = sub.add_parser("generate", help="Generate clean/noisy pairs")
    p_gen.add_argument("--clean-dir", required=True)
    p_gen.add_argument("--noise-dir", required=True)
    p_gen.add_argument("--out-dir", required=True)
    p_gen.add_argument("--sample-rate", type=int, default=16_000)
    p_gen.add_argument("--clip-seconds", type=float, default=4.0)
    p_gen.add_argument("--seed", type=int, default=42)

    # train
    p_train = sub.add_parser("train", help="Train the prosthesis SNN")
    p_train.add_argument("--data-dir", default="data/prosthesis/generated")
    p_train.add_argument("--output-dir", default="data/prosthesis/training_runs")
    p_train.add_argument("--epochs", type=int, default=40)
    p_train.add_argument("--batch-size", type=int, default=16)
    p_train.add_argument("--lr", type=float, default=3e-4)
    p_train.add_argument("--device", default="")

    # export
    p_export = sub.add_parser("export", help="Export model to .nmap")
    p_export.add_argument("--checkpoint", required=True)
    p_export.add_argument("--output", default="model_export.nmap")
    p_export.add_argument("--bits", type=int, default=4)
    p_export.add_argument("--device", default="cpu")

    args = parser.parse_args()
    {
        "bootstrap": cmd_bootstrap,
        "generate": cmd_generate,
        "train": cmd_train,
        "export": cmd_export,
    }[args.command](args)


if __name__ == "__main__":
    main()
