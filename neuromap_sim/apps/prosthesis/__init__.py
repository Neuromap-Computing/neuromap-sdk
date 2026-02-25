"""Prosthesis dataset bootstrap + generation app exports."""

from neuromap_sim.apps.prosthesis.config import (
    GammatoneFeatureConfig,
    ProsthesisBootstrapConfig,
    ProsthesisDatasetConfig,
)
from neuromap_sim.apps.prosthesis.data_bootstrap import bootstrap_prosthesis_raw_data
from neuromap_sim.apps.prosthesis.data_generation import generate_prosthesis_dataset
from neuromap_sim.apps.prosthesis.pipeline import ProsthesisApp

__all__ = [
    "ProsthesisApp",
    "GammatoneFeatureConfig",
    "ProsthesisBootstrapConfig",
    "ProsthesisDatasetConfig",
    "bootstrap_prosthesis_raw_data",
    "generate_prosthesis_dataset",
]

