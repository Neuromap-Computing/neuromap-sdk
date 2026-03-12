"""Live spike monitoring with callbacks and visualization.

The :class:`SpikeMonitor` captures spike events streamed from the board
and provides callbacks, collection, and plotting utilities.

Example::

    with board.start_monitor() as mon:
        mon.on_spike(lambda e: print(e))
        events = mon.collect(duration=2.0)
        mon.raster_plot(events)
"""

from __future__ import annotations

import struct
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

# ---------------------------------------------------------------------------
# SpikeEvent
# ---------------------------------------------------------------------------


@dataclass
class SpikeEvent:
    """A single spike event from the board.

    Attributes:
        timestamp_us: Event timestamp in microseconds.
        layer: Layer index where the spike occurred.
        neuron: Neuron index within the layer.
        event_type: Type of event (``"spike"`` or ``"membrane_cross"``).
    """

    timestamp_us: int
    layer: int
    neuron: int
    event_type: str = "spike"


_EVENT_TYPE_MAP = {
    0: "spike",
    1: "membrane_cross",
}


# ---------------------------------------------------------------------------
# SpikeMonitor
# ---------------------------------------------------------------------------


class SpikeMonitor:
    """Live spike monitoring with callbacks and visualization.

    Do not instantiate directly — use :meth:`Board.start_monitor`.

    Args:
        board: The board to monitor.
    """

    def __init__(self, board: Any) -> None:
        self._board = board
        self._callbacks: list[Callable[[SpikeEvent], None]] = []
        self._events: list[SpikeEvent] = []
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def on_spike(self, callback: Callable[[SpikeEvent], None]) -> None:
        """Register a callback for spike events.

        Args:
            callback: Called with each :class:`SpikeEvent` as it arrives.
        """
        self._callbacks.append(callback)

    def collect(self, duration: float) -> list[SpikeEvent]:
        """Collect spike events for a fixed duration.

        Args:
            duration: Collection duration in seconds.

        Returns:
            List of collected :class:`SpikeEvent` objects.
        """
        self._events.clear()
        self._start_background()
        time.sleep(duration)
        self._stop_background()
        with self._lock:
            return list(self._events)

    def raster_plot(
        self,
        events: list[SpikeEvent] | None = None,
        *,
        layer: int | None = None,
        title: str = "Spike Raster Plot",
    ) -> Any:
        """Generate a raster plot of spike events.

        Args:
            events: Events to plot.  Defaults to internally collected events.
            layer: Optional layer filter.
            title: Plot title.

        Returns:
            A matplotlib :class:`~matplotlib.figure.Figure`.

        Raises:
            ImportError: If matplotlib is not installed.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise ImportError(
                "matplotlib is required for raster plots. "
                "Install it with: pip install neuromap[monitor]"
            ) from exc

        if events is None:
            events = list(self._events)

        if layer is not None:
            events = [e for e in events if e.layer == layer]

        if not events:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.set_title(title)
            ax.set_xlabel("Time (us)")
            ax.set_ylabel("Neuron")
            return fig

        times = [e.timestamp_us for e in events]
        neurons = [e.layer * 100 + e.neuron for e in events]  # offset by layer
        colors = [e.layer for e in events]

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(times, neurons, c=colors, s=2, cmap="tab10")
        ax.set_title(title)
        ax.set_xlabel("Time (us)")
        ax.set_ylabel("Layer.Neuron")
        return fig

    def firing_rate(self, events: list[SpikeEvent] | None = None) -> dict[int, float]:
        """Compute per-layer firing rates.

        Args:
            events: Events to analyze.  Defaults to internally collected.

        Returns:
            Dict mapping layer index to firing rate (spikes/second).
        """
        if events is None:
            events = list(self._events)

        if not events:
            return {}

        # Group by layer
        layer_events: dict[int, list[SpikeEvent]] = {}
        for e in events:
            layer_events.setdefault(e.layer, []).append(e)

        # Compute rates
        all_times = [e.timestamp_us for e in events]
        duration_s = (max(all_times) - min(all_times)) / 1_000_000.0
        if duration_s <= 0:
            return {layer: 0.0 for layer in layer_events}

        return {layer: len(evts) / duration_s for layer, evts in layer_events.items()}

    def stop(self) -> None:
        """Stop monitoring and send MONITOR_STOP to the board."""
        self._stop_background()
        from neuromap.protocol import Cmd

        try:
            self._board._send_cmd(Cmd.MONITOR_STOP, b"", expect=Cmd.MONITOR_STOP_ACK)
        except Exception:
            pass

    # -- Context manager ----------------------------------------------------

    def __enter__(self) -> SpikeMonitor:
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()

    # -- Internal -----------------------------------------------------------

    def _start_background(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _stop_background(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _read_loop(self) -> None:
        from neuromap.protocol import Cmd, PacketCodec

        codec = PacketCodec()
        while self._running:
            try:
                raw = self._board._transport.read(4106, timeout_ms=100)
                if not raw:
                    continue
                packets = codec.feed(raw)
                for pkt in packets:
                    if pkt.cmd == Cmd.SPIKE_EVENT and len(pkt.payload) >= 7:
                        ts_us, layer_idx, neuron_idx, evt_type = struct.unpack_from(
                            "<IBBB", pkt.payload
                        )
                        event = SpikeEvent(
                            timestamp_us=ts_us,
                            layer=layer_idx,
                            neuron=neuron_idx,
                            event_type=_EVENT_TYPE_MAP.get(evt_type, "spike"),
                        )
                        with self._lock:
                            self._events.append(event)
                        for cb in self._callbacks:
                            try:
                                cb(event)
                            except Exception:
                                pass
            except Exception:
                if not self._running:
                    break
                time.sleep(0.01)
