"""Tests for Board API with mock backend."""

import numpy as np
import pytest
import torch
from neuromap import Board, BoardInfo, Exporter, Network, chips


class TestBoardMock:
    """Board creation and info with mock transport."""

    def test_mock_creates(self) -> None:
        board = Board.mock()
        assert board.is_connected
        assert board.is_mock
        board.close()

    def test_mock_info(self) -> None:
        board = Board.mock()
        info = board.info
        assert info is not None
        assert isinstance(info, BoardInfo)
        assert info.board_id == 1
        assert info.fw_version == (1, 0, 0)
        assert info.chip_type == "NeuroSoC-v1"
        assert info.status == "idle"
        assert info.serial_port == "mock"
        board.close()

    def test_mock_context_manager(self) -> None:
        with Board.mock() as board:
            assert board.is_connected
        assert not board.is_connected

    def test_mock_repr(self) -> None:
        board = Board.mock()
        assert "mock" in repr(board)
        board.close()

    def test_mock_with_custom_chip(self) -> None:
        from neuromap.chip import ChipSpec

        chip = ChipSpec(name="test", layers=(8, 8), weight_bits=4)
        board = Board.mock(chip)
        assert board.is_connected
        board.close()


class TestBoardProgram:
    """Programming and deploy tests."""

    def test_program_with_nmap_bytes(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        exporter = Exporter(net)
        exporter.quantize()
        nmap_bytes = exporter.to_bytes()

        with Board.mock() as board:
            board.program(nmap_bytes)
            assert board._programmed

    def test_deploy_from_network(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        with Board.mock() as board:
            net.deploy(board)
            assert board._programmed

    def test_deploy_with_verify(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        with Board.mock() as board:
            net.deploy(board, verify=True)

    def test_deploy_without_verify(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        with Board.mock() as board:
            net.deploy(board, verify=False)

    def test_verify_before_program_raises(self) -> None:
        with Board.mock() as board:
            with pytest.raises(RuntimeError, match="not programmed"):
                board.verify()


class TestBoardInference:
    """Inference tests with mock board."""

    def test_inject_returns_output(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        with Board.mock() as board:
            net.deploy(board)
            board.start_inference()
            spikes = np.zeros(16, dtype=np.float32)
            spikes[0] = 1.0
            output = board.inject(spikes)
            assert isinstance(output, np.ndarray)
            assert output.shape[0] == 16  # output_size
            board.stop_inference()

    def test_inject_with_torch_tensor(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        with Board.mock() as board:
            net.deploy(board)
            board.start_inference()
            spikes = torch.zeros(16)
            spikes[0] = 1.0
            output = board.inject(spikes)
            assert isinstance(output, np.ndarray)
            board.stop_inference()

    def test_inject_batch(self) -> None:
        net = Network(chips.NEUROSOC_V1)
        with Board.mock() as board:
            net.deploy(board)
            board.start_inference()
            spikes = np.zeros((5, 16), dtype=np.float32)
            spikes[0, 0] = 1.0
            output = board.inject_batch(spikes)
            assert isinstance(output, np.ndarray)
            board.stop_inference()


class TestBoardSystem:
    """System commands."""

    def test_reset(self) -> None:
        with Board.mock() as board:
            board.reset()

    def test_set_led(self) -> None:
        with Board.mock() as board:
            board.set_led(0, True)
            board.set_led(0, False)

    def test_start_monitor(self) -> None:
        with Board.mock() as board:
            mon = board.start_monitor()
            assert mon is not None
            mon.stop()
