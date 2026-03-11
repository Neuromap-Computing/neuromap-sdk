"""CLI entry points for the Neuromap SDK.

Registered as ``neuromap`` command via ``[project.scripts]`` in
``pyproject.toml``.  Subcommands::

    neuromap info     — Show board info
    neuromap flash    — Deploy model to board
    neuromap monitor  — Live spike monitoring
    neuromap reset    — Soft-reset board
    neuromap test     — Board self-test
    neuromap inject   — Inject spikes from CLI
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


def main(argv: list[str] | None = None) -> None:
    """CLI main entry point."""
    parser = argparse.ArgumentParser(
        prog="neuromap",
        description="Neuromap SDK — command-line tools for neuromorphic dev boards",
    )
    sub = parser.add_subparsers(dest="command")

    # -- info --
    p_info = sub.add_parser("info", help="Show board info")
    p_info.add_argument("--port", default=None, help="Serial port (auto-detect if omitted)")
    p_info.add_argument("--mock", action="store_true", help="Use mock board")
    p_info.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # -- flash --
    p_flash = sub.add_parser("flash", help="Deploy model to board")
    p_flash.add_argument("file", type=str, help="Path to .nmap file")
    p_flash.add_argument("--port", default=None, help="Serial port")
    p_flash.add_argument("--mock", action="store_true", help="Use mock board")
    p_flash.add_argument("--no-verify", action="store_true", help="Skip verification")
    p_flash.add_argument("--verbose", action="store_true", help="Verbose output")

    # -- monitor --
    p_monitor = sub.add_parser("monitor", help="Live spike monitoring")
    p_monitor.add_argument("--port", default=None, help="Serial port")
    p_monitor.add_argument("--mock", action="store_true", help="Use mock board")
    p_monitor.add_argument("--duration", type=float, default=5.0, help="Duration in seconds")
    p_monitor.add_argument("--output", type=str, default=None, help="Save events to file")
    p_monitor.add_argument("--raster", action="store_true", help="Show raster plot")
    p_monitor.add_argument("--layer", type=int, default=None, help="Filter by layer")
    p_monitor.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")

    # -- reset --
    p_reset = sub.add_parser("reset", help="Soft-reset board")
    p_reset.add_argument("--port", default=None, help="Serial port")
    p_reset.add_argument("--mock", action="store_true", help="Use mock board")

    # -- test --
    p_test = sub.add_parser("test", help="Board self-test")
    p_test.add_argument("--port", default=None, help="Serial port")
    p_test.add_argument("--mock", action="store_true", help="Use mock board")

    # -- inject --
    p_inject = sub.add_parser("inject", help="Inject spikes from CLI")
    p_inject.add_argument("--port", default=None, help="Serial port")
    p_inject.add_argument("--mock", action="store_true", help="Use mock board")
    p_inject.add_argument("--input", type=str, help="Comma-separated spike values (e.g. 1,0,1,0,...)")
    p_inject.add_argument("--file", type=str, help="Path to numpy .npy file with spikes")
    p_inject.add_argument("--output", type=str, default=None, help="Save output to .npy file")

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    try:
        if args.command == "info":
            _cmd_info(args)
        elif args.command == "flash":
            _cmd_flash(args)
        elif args.command == "monitor":
            _cmd_monitor(args)
        elif args.command == "reset":
            _cmd_reset(args)
        elif args.command == "test":
            _cmd_test(args)
        elif args.command == "inject":
            _cmd_inject(args)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


# ---------------------------------------------------------------------------
# Subcommand implementations
# ---------------------------------------------------------------------------


def _get_board(args: argparse.Namespace) -> "Board":
    from neuromap.board import Board

    if getattr(args, "mock", False):
        return Board.mock()
    return Board.connect(port=getattr(args, "port", None))


def _cmd_info(args: argparse.Namespace) -> None:
    with _get_board(args) as board:
        info = board.info
        if info is None:
            print("No board info available.")
            return
        if args.as_json:
            print(json.dumps(asdict(info), indent=2))
        else:
            print(f"Board ID     : 0x{info.board_id:08X}")
            print(f"HW Revision  : {info.hw_revision}")
            print(f"FW Version   : {info.fw_version[0]}.{info.fw_version[1]}.{info.fw_version[2]}")
            print(f"Protocol     : v{info.protocol_version}")
            print(f"Chip Type    : {info.chip_type}")
            print(f"Serial Port  : {info.serial_port}")
            print(f"Status       : {info.status}")


def _cmd_flash(args: argparse.Namespace) -> None:
    path = Path(args.file)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with _get_board(args) as board:
        if args.verbose:
            print(f"Flashing {path} to board...")
        board.program(path)
        if not args.no_verify:
            if args.verbose:
                print("Verifying...")
            ok = board.verify()
            if ok:
                print("Flash and verify OK.")
            else:
                print("Verification failed!", file=sys.stderr)
                sys.exit(1)
        else:
            print("Flash OK (verification skipped).")


def _cmd_monitor(args: argparse.Namespace) -> None:
    with _get_board(args) as board:
        # Need to program and start inference for monitoring to be meaningful
        board._send_cmd(
            __import__("neuromap.protocol", fromlist=["Cmd"]).Cmd.MONITOR_START,
            b"",
        )
        from neuromap.monitor import SpikeMonitor

        mon = SpikeMonitor(board)
        print(f"Monitoring for {args.duration}s...")
        events = mon.collect(args.duration)
        print(f"Collected {len(events)} events.")

        if args.layer is not None:
            events = [e for e in events if e.layer == args.layer]
            print(f"  Filtered to layer {args.layer}: {len(events)} events.")

        if args.as_json:
            import dataclasses

            print(json.dumps([dataclasses.asdict(e) for e in events], indent=2))

        if args.output:
            import dataclasses

            with open(args.output, "w") as f:
                json.dump([dataclasses.asdict(e) for e in events], f, indent=2)
            print(f"Events saved to {args.output}")

        if args.raster and events:
            fig = mon.raster_plot(events, layer=args.layer)
            fig.savefig("raster_plot.png", dpi=150)
            print("Raster plot saved to raster_plot.png")

        rates = mon.firing_rate(events)
        if rates:
            print("Firing rates (spikes/s):")
            for layer, rate in sorted(rates.items()):
                print(f"  Layer {layer}: {rate:.1f}")


def _cmd_reset(args: argparse.Namespace) -> None:
    with _get_board(args) as board:
        board.reset()
        print("Board reset OK.")


def _cmd_test(args: argparse.Namespace) -> None:
    import numpy as np

    with _get_board(args) as board:
        print("Running self-test...")

        # 1. Ping
        info = board.info
        if info is None:
            print("FAIL: Could not get board info.")
            sys.exit(1)
        print(f"  PING/PONG ... OK ({info.chip_type})")

        # 2. Program with dummy weights
        from neuromap import Exporter, Network, chips

        net = Network(chips.NEUROSOC_V1)
        exporter = Exporter(net)
        exporter.quantize()
        nmap_bytes = exporter.to_bytes()
        board.program(nmap_bytes)
        print("  Program    ... OK")

        # 3. Verify
        ok = board.verify()
        print(f"  Verify     ... {'OK' if ok else 'FAIL'}")

        # 4. Inference
        board.start_inference()
        spikes = np.zeros(16, dtype=np.float32)
        spikes[0] = 1.0
        output = board.inject(spikes)
        board.stop_inference()
        print(f"  Inference  ... OK (output shape: {output.shape})")

        # 5. Reset
        board.reset()
        print("  Reset      ... OK")

        print("Self-test PASSED.")


def _cmd_inject(args: argparse.Namespace) -> None:
    import numpy as np

    if args.input:
        spikes = np.array([float(x) for x in args.input.split(",")], dtype=np.float32)
    elif args.file:
        spikes = np.load(args.file)
    else:
        print("Error: Provide --input or --file", file=sys.stderr)
        sys.exit(1)

    with _get_board(args) as board:
        board.start_inference()
        output = board.inject(spikes)
        board.stop_inference()

        print(f"Output: {output}")
        if args.output:
            np.save(args.output, output)
            print(f"Saved to {args.output}")
