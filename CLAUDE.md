# neuro-sim — Neuromap Python SDK

Nx project `neuro-sim`; PyPI package `neuromap` (v0.1.4). Builds, trains,
quantizes and exports spiking neural networks targeting Neuromap neuromorphic
chips. Built on PyTorch + [snnTorch](https://snntorch.readthedocs.io/).

**This directory is mirrored to a public repo** (`Neuromap-Computing/neuromap-sdk`)
on every push to `main`, including commit messages. Keep SDK changes in their own
commits.

## Commands

```bash
npx nx run neuro-sim:test          # uv run pytest tests/ -v
npx nx run neuro-sim:lint          # ruff check
npx nx run neuro-sim:format        # ruff format
npx nx run neuro-sim:format-check  # CI gate
npx nx run neuro-sim:build         # uv build (sdist + wheel)
```

Direct, from `neuro-sim/`: `uv run pytest tests/test_network.py -v -k quantize`.

## Layout

```
src/neuromap/
  __init__.py      public API surface — the __all__ list is the contract
  chip.py          ChipSpec, NeuronParams, chips registry (chips.NEUROSOC_V1)
  network.py       Network — build/infer/save/load
  trainer.py       Trainer, TrainHistory — fit loop, LR schedule, early stop
  export.py        Exporter — quantize + write .nmap bundles
  convenience.py   encode_sensor_data / compile_model / quantize_and_export_to_spi
  _internal/       lif, dynamic_snn, encoding, quantization, spike, audio
tests/             one module per public module, plus conftest.py fixtures
examples/prosthesis/   end-to-end EMG prosthesis pipeline
docs/              MkDocs site → docs.neuromap.ca
```

## Conventions

- **Public vs internal is load-bearing.** Anything under `_internal/` may change
  without notice. To add a public symbol, export it from `__init__.py` *and* add
  it to `__all__` *and* document it in `README.md` + `docs/docs/api/`.
- `ChipSpec` and `NeuronParams` are **frozen dataclasses** — never mutate; build
  a new one. Both round-trip via `to_dict()` / `from_dict()`.
- The chip topology is fixed by the target: `chips.NEUROSOC_V1` is 48 LIF
  neurons, 512 4-bit synapses, **16 input pins**. Code and tests must respect
  those limits — they are the physical constraint the whole product exists for.
- `.nmap` is a ZIP: `manifest.json` (chip spec, topology, quantization metadata)
  plus per-layer arrays under `weights/`. Changing the format means bumping the
  manifest version and updating the firmware parser
  (`firmware/main/network/nmap_parser.c`) and `Exporter.load_nmap`.
- **Keep torch/snnTorch imports at module scope** in `_internal`, but avoid
  pulling heavy optional deps (scipy, matplotlib) into the import path of
  `neuromap/__init__.py` — import time is part of the SDK's UX.
- Ruff config is at the **repo root** `pyproject.toml`, not here.
- Every public function needs a **Google-style docstring** with types in the
  signature — `docs/` renders them through mkdocstrings, so a malformed
  docstring shows up as broken output on docs.neuromap.ca.

## Testing

`tests/` mirrors the module layout. New public behaviour needs a test in the
matching `test_<module>.py`. Prefer the small synthetic loaders in `conftest.py`
over downloading datasets — the suite must run offline and fast in CI.
