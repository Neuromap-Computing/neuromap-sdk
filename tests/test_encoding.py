"""Tests for spike encoding (encode_sensor_data)."""

from __future__ import annotations

import pytest
import torch
from neuromap import encode_sensor_data


class TestEncodeSensorData:
    def test_rate_encoding_shape(self) -> None:
        data = torch.rand(8, 16)
        spikes = encode_sensor_data(data, num_steps=50, method="rate")
        assert spikes.shape == (8, 50, 16)

    def test_rate_encoding_binary(self) -> None:
        data = torch.rand(4, 8)
        spikes = encode_sensor_data(data, num_steps=20, method="rate")
        unique = torch.unique(spikes)
        assert all(v in (0.0, 1.0) for v in unique.tolist())

    def test_latency_encoding_shape(self) -> None:
        data = torch.rand(4, 8)
        spikes = encode_sensor_data(data, num_steps=30, method="latency")
        assert spikes.shape == (4, 30, 8)

    def test_delta_encoding_shape(self) -> None:
        data = torch.rand(4, 20, 8)
        spikes = encode_sensor_data(data, method="delta")
        assert spikes.shape == (4, 20, 8)

    def test_default_method_is_rate(self) -> None:
        data = torch.rand(2, 4)
        spikes = encode_sensor_data(data, num_steps=10)
        assert spikes.shape == (2, 10, 4)

    def test_unknown_method_raises(self) -> None:
        data = torch.rand(2, 4)
        with pytest.raises(ValueError, match="Unknown encoding method"):
            encode_sensor_data(data, method="unknown")
