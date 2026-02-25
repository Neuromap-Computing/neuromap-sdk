"""Prosthesis app plugin entrypoint."""

from __future__ import annotations

from typing import Any

from neuromap_sim.apps.base import BaseApp
from neuromap_sim.apps.prosthesis.config import ProsthesisBootstrapConfig, ProsthesisDatasetConfig
from neuromap_sim.apps.prosthesis.data_bootstrap import bootstrap_prosthesis_raw_data
from neuromap_sim.apps.prosthesis.data_generation import generate_prosthesis_dataset


class ProsthesisApp(BaseApp):
    """App facade for dataset bootstrap and generation flows."""

    name = "prosthesis"

    def run(self, config: dict[str, Any], **kwargs: Any) -> dict[str, Any]:
        action = kwargs.get("action")
        if action == "bootstrap":
            cfg = (
                config
                if isinstance(config, ProsthesisBootstrapConfig)
                else ProsthesisBootstrapConfig(**config)
            )
            result = bootstrap_prosthesis_raw_data(cfg)
            return {"app": self.name, "action": "bootstrap", "config": cfg.to_dict(), **result}

        if action == "generate":
            cfg = (
                config
                if isinstance(config, ProsthesisDatasetConfig)
                else ProsthesisDatasetConfig(**config)
            )
            clean_dir = kwargs.get("clean_dir")
            noise_dir = kwargs.get("noise_dir")
            out_dir = kwargs.get("out_dir")
            if not clean_dir or not noise_dir or not out_dir:
                raise ValueError("clean_dir, noise_dir and out_dir are required for generate action.")
            result = generate_prosthesis_dataset(
                clean_dir=clean_dir,
                noise_dir=noise_dir,
                out_dir=out_dir,
                config=cfg,
            )
            return {"app": self.name, "action": "generate", "config": cfg.to_dict(), **result}

        raise ValueError("Unknown action. Expected one of: bootstrap, generate.")

