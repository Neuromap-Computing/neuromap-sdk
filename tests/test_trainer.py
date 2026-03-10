"""Tests for neuromap.trainer (Trainer, TrainHistory)."""

from __future__ import annotations

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from neuromap import Network, Trainer


def _make_loader(input_size: int, output_size: int, n: int = 20, time_steps: int = 5) -> DataLoader:
    """Build a tiny DataLoader of (x, y) tuples for testing."""
    x = torch.rand(n, time_steps, input_size)
    y = torch.rand(n, output_size)
    return DataLoader(TensorDataset(x, y), batch_size=4)


class TestTrainer:
    def test_fit_basic(self) -> None:
        net = Network.from_topology([16, 8, 4], use_decoder=False)
        loader = _make_loader(16, 4, n=12)
        trainer = Trainer(net, lr=1e-3, epochs=3, device="cpu")
        history = trainer.fit(loader, verbose=False)
        assert len(history.epochs) == 3
        assert "train_loss" in history.epochs[0]

    def test_fit_with_validation_and_early_stop(self) -> None:
        net = Network.from_topology([16, 8, 4], use_decoder=False)
        train_loader = _make_loader(16, 4, n=12)
        val_loader = _make_loader(16, 4, n=8)
        trainer = Trainer(
            net, lr=1e-3, epochs=100, device="cpu", early_stop_patience=3,
        )
        history = trainer.fit(train_loader, val_loader, verbose=False)
        assert len(history.epochs) <= 100
        assert "val_loss" in history.epochs[0]

    def test_evaluate(self) -> None:
        net = Network.from_topology([16, 8, 4], use_decoder=False)
        loader = _make_loader(16, 4, n=8)
        trainer = Trainer(net, device="cpu")
        metrics = trainer.evaluate(loader)
        assert "loss" in metrics
        assert isinstance(metrics["loss"], float)

    def test_invalid_network_type(self) -> None:
        with pytest.raises(TypeError, match="Network"):
            Trainer("not_a_network")  # type: ignore[arg-type]

    def test_sequence_forward(self) -> None:
        net = Network.from_topology([16, 8, 4], use_decoder=True)
        n, ts = 12, 5
        x = torch.rand(n, ts, 16)
        y = torch.rand(n, ts, 4)
        loader = DataLoader(TensorDataset(x, y), batch_size=4)
        trainer = Trainer(net, lr=1e-3, epochs=2, device="cpu", use_sequence_forward=True)
        history = trainer.fit(loader, verbose=False)
        assert len(history.epochs) == 2
