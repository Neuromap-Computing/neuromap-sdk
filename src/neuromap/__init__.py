"""Neuromap SDK — program neuromorphic chips with Python.

The public API exposes the following symbols plus :data:`__version__`::

    from neuromap import (
        Network, Trainer, Exporter, ChipSpec, NeuronParams, chips,
        TrainHistory, Board, BoardInfo,
        encode_sensor_data, compile_model, quantize_and_export_to_spi,
    )
"""

from neuromap._version import __version__
from neuromap.board import Board, BoardInfo
from neuromap.chip import ChipSpec, NeuronParams, chips
from neuromap.convenience import compile_model, encode_sensor_data, quantize_and_export_to_spi
from neuromap.export import Exporter
from neuromap.network import Network
from neuromap.trainer import Trainer, TrainHistory

__all__ = [
    "__version__",
    "Board",
    "BoardInfo",
    "ChipSpec",
    "compile_model",
    "encode_sensor_data",
    "Exporter",
    "Network",
    "NeuronParams",
    "quantize_and_export_to_spi",
    "Trainer",
    "TrainHistory",
    "chips",
]
