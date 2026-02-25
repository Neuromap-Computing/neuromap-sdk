"""Core SNN primitives."""

from neuromap_sim.core.lif import LIFLayer
from neuromap_sim.core.quantization import quantize_model_weights, quantize_weights_4bit
from neuromap_sim.core.spike import SurrogateHeaviside, spike_function

__all__ = [
    "LIFLayer",
    "SurrogateHeaviside",
    "spike_function",
    "quantize_model_weights",
    "quantize_weights_4bit",
]
