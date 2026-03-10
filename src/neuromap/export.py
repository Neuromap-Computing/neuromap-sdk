"""Export Neuromap networks to chip-compatible artifacts.

The :class:`Exporter` quantizes model weights to match the hardware DAC
bit-width and packages them into a ``.nmap`` archive that can be loaded
onto a Neuromap chip.

Example::

    from neuromap import Exporter, Network, chips

    net = Network(chips.NEUROSOC_V1)
    Exporter(net).quantize().save("model.nmap")
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

from neuromap._internal.quantization import quantize_model_weights


class Exporter:
    """Quantize and serialise a :class:`~neuromap.network.Network`.

    The primary output format is ``.nmap`` — a simple ZIP archive
    containing:

    * ``manifest.json`` — chip spec, topology, quantization metadata.
    * ``weights/layer_<i>_weight.npy`` — per-layer quantized weight
      matrices as integer numpy arrays.
    * ``weights/layer_<i>_bias.npy`` — per-layer bias vectors.

    Args:
        network: A :class:`~neuromap.network.Network` instance whose
            underlying model will be exported.
    """

    def __init__(self, network: Any) -> None:
        from neuromap.network import Network  # deferred for cycle avoidance

        if not isinstance(network, Network):
            raise TypeError("Exporter expects a neuromap.Network instance.")
        self._network = network
        self._quantized = False
        self._bits: int | None = None

    def quantize(self, bits: int | None = None) -> Exporter:
        """Apply post-training quantization to the model weights.

        Args:
            bits: Bit-width for quantization.  Defaults to the chip's
                ``weight_bits`` setting.

        Returns:
            This :class:`Exporter` instance for chaining.
        """
        actual_bits = bits if bits is not None else self._network.chip.weight_bits
        quantize_model_weights(self._network.model, bits=actual_bits)
        self._quantized = True
        self._bits = actual_bits
        return self

    def to_weight_map(self) -> dict[str, np.ndarray]:
        """Extract per-layer quantized weight arrays.

        Returns:
            Dictionary mapping ``"layer_<i>_weight"`` (and optionally
            ``"layer_<i>_bias"``) to numpy arrays.  When :meth:`quantize`
            was called, weight values are clipped integer arrays.
        """
        bits = self._bits or self._network.chip.weight_bits
        qmax = 2 ** (bits - 1) - 1

        weight_map: dict[str, np.ndarray] = {}
        state = self._network.model.state_dict()
        for key, tensor in state.items():
            if "weight" not in key and "bias" not in key:
                continue
            arr = tensor.detach().cpu().numpy()
            if self._quantized and "weight" in key:
                scale = float(np.max(np.abs(arr))) / qmax if np.max(np.abs(arr)) > 0 else 1.0
                int_arr = np.clip(np.round(arr / scale), -qmax, qmax).astype(np.int8)
                weight_map[key] = int_arr
            else:
                weight_map[key] = arr
        return weight_map

    def save(self, path: str | Path) -> None:
        """Save a ``.nmap`` bundle (ZIP archive with manifest + weights).

        Args:
            path: Destination file path (conventionally ``*.nmap``).
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        manifest = self._build_manifest()
        weight_map = self.to_weight_map()

        with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.writestr("manifest.json", json.dumps(manifest, indent=2))
            for name, arr in weight_map.items():
                buf = io.BytesIO()
                np.save(buf, arr, allow_pickle=False)
                zf.writestr(f"weights/{name}.npy", buf.getvalue())

    @classmethod
    def load_nmap(cls, path: str | Path, *, device: str | torch.device = "cpu") -> Any:
        """Load a ``.nmap`` bundle and reconstruct a :class:`Network`.

        Args:
            path: Path to the ``.nmap`` file.
            device: Device to load the model onto.

        Returns:
            A :class:`~neuromap.network.Network` instance with the
            stored weights loaded.
        """
        from neuromap.chip import ChipSpec
        from neuromap.network import Network

        path = Path(path)
        with zipfile.ZipFile(path, "r") as zf:
            manifest = json.loads(zf.read("manifest.json"))
            chip = ChipSpec.from_dict(manifest["chip_spec"])
            use_decoder = manifest.get("use_decoder", True)
            net = Network(chip, use_decoder=use_decoder)

            bits = manifest.get("weight_bits", chip.weight_bits)
            qmax = 2 ** (bits - 1) - 1

            state_dict = net.model.state_dict()
            for name in list(state_dict.keys()):
                npy_path = f"weights/{name}.npy"
                if npy_path not in zf.namelist():
                    continue
                buf = io.BytesIO(zf.read(npy_path))
                arr = np.load(buf, allow_pickle=False)

                if "weight" in name and manifest.get("quantized", False):
                    state_dict[name] = torch.from_numpy(arr.astype(np.float32))
                else:
                    state_dict[name] = torch.from_numpy(arr.astype(np.float32))

            net.model.load_state_dict(state_dict)
            net.to(device)
        return net

    def _build_manifest(self) -> dict[str, Any]:
        chip = self._network.chip
        return {
            "format": "nmap-v1",
            "chip_spec": chip.to_dict(),
            "layers": list(chip.layers),
            "weight_bits": self._bits or chip.weight_bits,
            "quantized": self._quantized,
            "use_decoder": self._network._use_decoder,
        }
