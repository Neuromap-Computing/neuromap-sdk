"""Neuromap SDK — program neuromorphic chips with Python.

The public API exposes the following symbols plus :data:`__version__`::

    from neuromap import (
        Network, Trainer, Exporter, ChipSpec, NeuronParams, chips,
        TrainHistory, Board, BoardInfo,
        encode_sensor_data, compile_model, quantize_and_export_to_spi,
    )
"""

from neuromap._version import __version__
from neuromap.aec import AdaptiveEventizationConverter
from neuromap.board import Board, BoardInfo
from neuromap.chip import ChipSpec, NeuronParams, chips

try:
    from neuromap.convenience import compile_model, encode_sensor_data, quantize_and_export_to_spi
except ModuleNotFoundError as exc:  # Optional dependency path; keep core package importable.
    _convenience_import_error = exc

    def _missing_convenience(*args: object, **kwargs: object) -> None:
        raise ModuleNotFoundError(
            "neuromap convenience helpers require the optional 'snntorch' dependency."
        ) from _convenience_import_error

    compile_model = _missing_convenience
    encode_sensor_data = _missing_convenience
    quantize_and_export_to_spi = _missing_convenience

try:
    from neuromap.export import Exporter
except ModuleNotFoundError as exc:  # Optional dependency path; keep core package importable.
    _export_import_error = exc

    def _missing_exporter(*args: object, **kwargs: object) -> None:
        raise ModuleNotFoundError(
            "neuromap.export requires optional ML dependencies that are not installed."
        ) from _export_import_error

    Exporter = _missing_exporter

try:
    from neuromap.network import Network
except ModuleNotFoundError as exc:  # Optional dependency path; keep core package importable.
    _network_import_error = exc

    def _missing_network(*args: object, **kwargs: object) -> None:
        raise ModuleNotFoundError(
            "neuromap.network requires optional ML dependencies that are not installed."
        ) from _network_import_error

    Network = _missing_network

try:
    from neuromap.trainer import Trainer, TrainHistory
except ModuleNotFoundError as exc:  # Optional dependency path; keep core package importable.
    _trainer_import_error = exc

    def _missing_trainer(*args: object, **kwargs: object) -> None:
        raise ModuleNotFoundError(
            "neuromap.trainer requires optional ML dependencies that are not installed."
        ) from _trainer_import_error

    Trainer = _missing_trainer
    TrainHistory = _missing_trainer

__all__ = [
    "__version__",
    "Board",
    "BoardInfo",
    "ChipSpec",
    "AdaptiveEventizationConverter",
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
