"""Reusable training loop for Neuromap SNN networks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


@dataclass
class TrainHistory:
    """Container for per-epoch training metrics."""

    epochs: list[dict[str, float]] = field(default_factory=list)

    def append(self, row: dict[str, float]) -> None:
        self.epochs.append(row)

    def to_list(self) -> list[dict[str, float]]:
        return list(self.epochs)


class Trainer:
    """High-level training driver for :class:`~neuromap_sim.sdk.network.Network`.

    Encapsulates gradient clipping, LR scheduling with warmup, early
    stopping, and checkpointing — mirroring the logic previously in
    ``scripts/train_prosthesis_snn.py`` but in a reusable form.

    Parameters
    ----------
    network:
        A :class:`~neuromap_sim.sdk.network.Network` instance.
    lr:
        Initial learning rate.
    epochs:
        Maximum number of training epochs.
    device:
        Torch device string (e.g. ``"cpu"``, ``"cuda"``).
    optimizer:
        Optimizer class name: ``"adam"`` (default) or ``"sgd"``.
    weight_decay:
        L2 regularisation weight.
    scheduler:
        LR scheduler type: ``"plateau"`` (default), ``"cosine"``, or ``"none"``.
    scheduler_factor:
        Factor by which LR is reduced (ReduceLROnPlateau).
    scheduler_patience:
        Epochs of no improvement before LR reduction.
    min_lr:
        Minimum learning rate for the scheduler.
    lr_warmup_epochs:
        Number of epochs for linear LR warmup from 0 to *lr*.
    early_stop_patience:
        Stop training after this many epochs without val-loss improvement.
    early_stop_min_delta:
        Minimum val-loss decrease to count as improvement.
    grad_clip_norm:
        Max gradient norm for clipping (``<=0`` disables).
    loss_fn:
        Optional custom loss function ``(preds, targets) -> scalar``.
        Defaults to MSE.
    use_sequence_forward:
        If ``True``, use ``model.forward_sequence`` for frame-level loss
        instead of rate-coded ``model.forward``.
    """

    def __init__(
        self,
        network: Any,  # avoid circular import – typed as sdk.Network at runtime
        *,
        lr: float = 3e-4,
        epochs: int = 40,
        device: str = "cpu",
        optimizer: str = "adam",
        weight_decay: float = 1e-6,
        scheduler: str = "plateau",
        scheduler_factor: float = 0.5,
        scheduler_patience: int = 2,
        min_lr: float = 1e-6,
        lr_warmup_epochs: int = 3,
        early_stop_patience: int = 8,
        early_stop_min_delta: float = 1e-4,
        grad_clip_norm: float = 1.0,
        loss_fn: Callable[..., torch.Tensor] | None = None,
        use_sequence_forward: bool = False,
    ) -> None:
        from neuromap_sim.sdk.network import Network  # deferred to avoid cycle

        if not isinstance(network, Network):
            raise TypeError("Trainer expects a neuromap_sim.sdk.Network instance.")

        self.network = network
        self.lr = lr
        self.epochs = epochs
        self.device = torch.device(device)
        self._optimizer_name = optimizer.lower()
        self.weight_decay = weight_decay
        self._scheduler_name = scheduler.lower()
        self.scheduler_factor = scheduler_factor
        self.scheduler_patience = scheduler_patience
        self.min_lr = min_lr
        self.lr_warmup_epochs = max(0, lr_warmup_epochs)
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.grad_clip_norm = grad_clip_norm
        self.use_sequence_forward = use_sequence_forward

        self._loss_fn = loss_fn or self._default_loss

        # Will be initialised in fit()
        self._optim: torch.optim.Optimizer | None = None
        self._sched: Any | None = None

    # -- public API ---------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        *,
        verbose: bool = True,
    ) -> TrainHistory:
        """Train the network and return a :class:`TrainHistory`.

        Each element in *train_loader* / *val_loader* must yield either:

        * A ``(x, y)`` tuple of tensors, **or**
        * An object with ``.x`` and ``.y`` attributes (like the existing
          ``Batch`` dataclass from the prosthesis training script).
        """
        model = self.network.model
        model.to(self.device)

        self._optim = self._build_optimizer(model)
        self._sched = self._build_scheduler(self._optim)

        history = TrainHistory()
        best_val_loss = float("inf")
        best_state_dict: dict[str, Any] | None = None
        epochs_without_improvement = 0

        for epoch in range(1, self.epochs + 1):
            # LR warmup
            if self.lr_warmup_epochs > 0 and epoch <= self.lr_warmup_epochs:
                warmup_factor = epoch / self.lr_warmup_epochs
                for g in self._optim.param_groups:
                    g["lr"] = self.lr * warmup_factor

            train_loss = self._train_one_epoch(model, train_loader)

            val_loss: float | None = None
            if val_loader is not None:
                val_loss = self._evaluate_loss(model, val_loader)

            # Scheduler step (only after warmup)
            if (
                self._sched is not None
                and epoch > self.lr_warmup_epochs
                and val_loss is not None
            ):
                if self._scheduler_name == "plateau":
                    self._sched.step(val_loss)
                else:
                    self._sched.step()

            current_lr = float(self._optim.param_groups[0]["lr"])
            row: dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": train_loss,
                "learning_rate": current_lr,
            }
            if val_loss is not None:
                row["val_loss"] = val_loss

            history.append(row)

            if verbose:
                parts = [f"epoch={epoch:03d}", f"train_loss={train_loss:.4f}"]
                if val_loss is not None:
                    parts.append(f"val_loss={val_loss:.4f}")
                parts.append(f"lr={current_lr:.2e}")
                print("  ".join(parts))

            # Early stopping
            if val_loss is not None:
                if val_loss < (best_val_loss - self.early_stop_min_delta):
                    best_val_loss = val_loss
                    best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement >= self.early_stop_patience:
                    if verbose:
                        print(
                            f"Early stopping at epoch {epoch}: "
                            f"no val_loss improvement for {self.early_stop_patience} epochs."
                        )
                    break

        # Restore best weights
        if best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        return history

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader) -> dict[str, float]:
        """Evaluate the network and return metric dict (at minimum ``loss``)."""
        model = self.network.model
        model.to(self.device)
        loss_val = self._evaluate_loss(model, dataloader)
        return {"loss": loss_val}

    # -- internals ----------------------------------------------------------------

    def _train_one_epoch(self, model: nn.Module, loader: DataLoader) -> float:
        model.train()
        assert self._optim is not None
        running_loss = 0.0
        batches = 0
        for batch in loader:
            x, y = self._unpack_batch(batch)
            x = x.to(self.device)
            y = y.to(self.device)

            self._optim.zero_grad(set_to_none=True)
            preds = self._forward(model, x)
            loss = self._loss_fn(preds, y)
            loss.backward()

            if self.grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=self.grad_clip_norm)
            self._optim.step()

            running_loss += float(loss.item())
            batches += 1

        return running_loss / max(1, batches)

    @torch.no_grad()
    def _evaluate_loss(self, model: nn.Module, loader: DataLoader) -> float:
        model.eval()
        running_loss = 0.0
        batches = 0
        for batch in loader:
            x, y = self._unpack_batch(batch)
            x = x.to(self.device)
            y = y.to(self.device)
            preds = self._forward(model, x)
            loss = self._loss_fn(preds, y)
            running_loss += float(loss.item())
            batches += 1
        return running_loss / max(1, batches)

    def _forward(self, model: nn.Module, x: torch.Tensor) -> torch.Tensor:
        if self.use_sequence_forward and hasattr(model, "forward_sequence"):
            return model.forward_sequence(x)
        return model(x)

    # -- helpers ------------------------------------------------------------------

    @staticmethod
    def _unpack_batch(batch: Any) -> tuple[torch.Tensor, torch.Tensor]:
        """Accept (x, y) tuples *or* objects with .x / .y attributes."""
        if isinstance(batch, (tuple, list)) and len(batch) >= 2:
            return batch[0], batch[1]
        if hasattr(batch, "x") and hasattr(batch, "y"):
            return batch.x, batch.y
        raise TypeError(
            "Trainer expects batches as (x, y) tuples or objects with .x/.y attributes."
        )

    @staticmethod
    def _default_loss(preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.mse_loss(preds, targets)

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        if self._optimizer_name == "sgd":
            return torch.optim.SGD(
                model.parameters(), lr=self.lr, weight_decay=self.weight_decay
            )
        return torch.optim.Adam(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )

    def _build_scheduler(self, optimizer: torch.optim.Optimizer) -> Any:
        if self._scheduler_name == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=self.scheduler_factor,
                patience=self.scheduler_patience,
                min_lr=self.min_lr,
            )
        if self._scheduler_name == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.epochs, eta_min=self.min_lr
            )
        return None

