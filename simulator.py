"""Backward-compatible entrypoint for the refactored Neuromap simulator."""

from __future__ import annotations

import json

from neuromap_sim.cli import run_demo as _run_demo
from neuromap_sim.core.lif import LIFLayer
from neuromap_sim.core.quantization import quantize_weights_4bit
from neuromap_sim.models.neuromap_snn import NeuromapSNN

__all__ = ["LIFLayer", "NeuromapSNN", "quantize_weights_4bit", "run_demo"]


def run_demo() -> None:
    """Run the classic smoke test with the refactored architecture."""
    demo = _run_demo()
    print("Neuromap SNN demo")
    print(f"input shape:  {tuple(demo['input_shape'])}")
    print(f"output shape: {tuple(demo['output_shape'])}")
    print(f"output dtype: {demo['dtype']}")
    print("4-bit quantization applied: OK")
    print(f"quantized output shape: {tuple(demo['quantized_output_shape'])}")
    print(json.dumps(demo, indent=2))


if __name__ == "__main__":
    run_demo()
