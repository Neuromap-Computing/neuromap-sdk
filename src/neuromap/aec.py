"""Adaptive eventization encoder for sparse, hardware-friendly spike generation."""

from __future__ import annotations

import numpy as np

__all__ = ["AdaptiveEventizationConverter"]


class AdaptiveEventizationConverter:
    """Adaptive Eventization Converter (AEC).

    Variants implemented:
    - ``prediction``: IIR prediction error > threshold.
    - ``derivative``: absolute derivative > threshold.
    - ``surprise``: normalized prediction error (error / sigma) > threshold.

    The converter maintains a target spike rate and adapts its threshold with a
    simple IIR homeostasis rule. All operations map cleanly to comparators,
    accumulators, and IIR filters.
    """

    def __init__(
        self,
        variant: str = "prediction",
        target_spike_rate: float = 0.05,
        alpha_pred: float = 0.01,
        alpha_thresh: float = 0.001,
        initial_threshold: float = 0.1,
        spike_rate_alpha: float = 0.01,
        local_var_alpha: float = 0.01,
        refractory: int = 0,
    ) -> None:
        self.variant = variant
        self.prediction = 0.0
        self.threshold = initial_threshold
        self.target_spike_rate = target_spike_rate
        self.alpha_pred = alpha_pred
        self.alpha_thresh = alpha_thresh
        self.spike_rate_est = 0.0
        self.spike_rate_alpha = spike_rate_alpha
        self.local_var_alpha = local_var_alpha
        self.running_var = 0.0
        self.refractory = refractory
        self.refractory_counter = 0
        self.last_x = 0.0

    def _update_prediction(self, x: float) -> float:
        error = x - self.prediction
        self.prediction += self.alpha_pred * error
        return error

    def _update_var(self, x: float) -> None:
        self.running_var = (
            (1 - self.local_var_alpha) * self.running_var
            + self.local_var_alpha * (x - self.prediction) ** 2
        )

    def _update_spike_rate(self, spike: int) -> None:
        self.spike_rate_est = (
            (1 - self.spike_rate_alpha) * self.spike_rate_est
            + self.spike_rate_alpha * spike
        )

    def step(self, x: float) -> tuple[int, int]:
        """Process one sample and return ``(spike, polarity)``."""
        if self.refractory_counter > 0:
            self.refractory_counter -= 1
            self._update_prediction(x)
            self._update_var(x)
            self._update_spike_rate(0)
            return 0, 0

        error = self._update_prediction(x)
        self._update_var(x)

        if self.variant == "prediction":
            measure = abs(error)
        elif self.variant == "derivative":
            measure = abs(x - self.last_x)
        elif self.variant == "surprise":
            sigma = max(1e-6, np.sqrt(self.running_var))
            measure = abs(error) / sigma
        else:
            measure = abs(error)

        spike = 0
        polarity = 0
        if measure > self.threshold:
            spike = 1
            polarity = 1 if error > 0 else -1
            self.refractory_counter = self.refractory

        self._update_spike_rate(spike)
        self.threshold += self.alpha_thresh * (self.spike_rate_est - self.target_spike_rate)
        self.threshold = max(self.threshold, 1e-6)
        self.last_x = x
        return spike, polarity

    def encode(self, signal: np.ndarray) -> list[tuple[int, int]]:
        """Encode a full signal into ``(t, polarity)`` events."""
        events: list[tuple[int, int]] = []
        for t, x in enumerate(signal):
            spike, polarity = self.step(float(x))
            if spike:
                events.append((t, polarity))
        return events

    def encode_with_trace(
        self, signal: np.ndarray
    ) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray, np.ndarray]:
        """Encode a signal and return events plus internal traces."""
        events: list[tuple[int, int]] = []
        preds: list[float] = []
        thresholds: list[float] = []
        spikes: list[int] = []

        for t, x in enumerate(signal):
            spike, polarity = self.step(float(x))
            if spike:
                events.append((t, polarity))
            preds.append(self.prediction)
            thresholds.append(self.threshold)
            spikes.append(spike)

        return events, np.array(preds), np.array(thresholds), np.array(spikes)

    def info(self) -> dict[str, float | str]:
        return {
            "variant": self.variant,
            "threshold": self.threshold,
            "spike_rate_est": self.spike_rate_est,
        }