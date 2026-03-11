"""Tests for SpikeMonitor."""

from neuromap import Board
from neuromap.monitor import SpikeEvent, SpikeMonitor


class TestSpikeEvent:
    """SpikeEvent dataclass."""

    def test_creation(self) -> None:
        event = SpikeEvent(timestamp_us=1000, layer=0, neuron=5)
        assert event.timestamp_us == 1000
        assert event.layer == 0
        assert event.neuron == 5
        assert event.event_type == "spike"

    def test_custom_event_type(self) -> None:
        event = SpikeEvent(
            timestamp_us=0, layer=1, neuron=2, event_type="membrane_cross"
        )
        assert event.event_type == "membrane_cross"


class TestSpikeMonitor:
    """SpikeMonitor with mock board."""

    def test_create_from_board(self) -> None:
        with Board.mock() as board:
            mon = board.start_monitor()
            assert isinstance(mon, SpikeMonitor)
            mon.stop()

    def test_context_manager(self) -> None:
        with Board.mock() as board:
            with board.start_monitor() as mon:
                assert isinstance(mon, SpikeMonitor)

    def test_on_spike_callback(self) -> None:
        events_received: list[SpikeEvent] = []
        with Board.mock() as board:
            mon = board.start_monitor()
            mon.on_spike(lambda e: events_received.append(e))
            mon.stop()

    def test_collect_returns_list(self) -> None:
        with Board.mock() as board:
            mon = board.start_monitor()
            # Short duration — mock doesn't generate spike events unprompted
            events = mon.collect(duration=0.1)
            assert isinstance(events, list)

    def test_firing_rate_empty(self) -> None:
        with Board.mock() as board:
            mon = board.start_monitor()
            rates = mon.firing_rate([])
            assert rates == {}
            mon.stop()

    def test_firing_rate_with_events(self) -> None:
        with Board.mock() as board:
            mon = board.start_monitor()
            events = [
                SpikeEvent(timestamp_us=0, layer=0, neuron=0),
                SpikeEvent(timestamp_us=1_000_000, layer=0, neuron=1),
                SpikeEvent(timestamp_us=500_000, layer=1, neuron=0),
            ]
            rates = mon.firing_rate(events)
            assert 0 in rates
            assert 1 in rates
            assert rates[0] == 2.0  # 2 spikes / 1 second
            mon.stop()
