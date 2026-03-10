"""Neuromap SDK — program neuromorphic chips with Python.

The public API exposes seven symbols plus :data:`__version__`::

    from neuromap import Network, Trainer, Exporter, ChipSpec, NeuronParams, chips, TrainHistory
"""

from neuromap._version import __version__
from neuromap.chip import ChipSpec, NeuronParams, chips
from neuromap.export import Exporter
from neuromap.network import Network
from neuromap.trainer import Trainer, TrainHistory

__all__ = [
    "__version__",
    "ChipSpec",
    "Exporter",
    "Network",
    "NeuronParams",
    "Trainer",
    "TrainHistory",
    "chips",
]
