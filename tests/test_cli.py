"""Tests for CLI entry points."""

import pytest
from neuromap.cli import main


class TestCLI:
    """CLI subcommand tests (using --mock flag)."""

    def test_no_args_prints_help(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            main([])
        assert exc_info.value.code == 1

    def test_info_mock(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["info", "--mock"])
        captured = capsys.readouterr()
        assert "NeuroSoC-v1" in captured.out

    def test_info_mock_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["info", "--mock", "--json"])
        captured = capsys.readouterr()
        assert '"chip_type"' in captured.out

    def test_reset_mock(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["reset", "--mock"])
        captured = capsys.readouterr()
        assert "reset ok" in captured.out.lower()

    def test_test_mock(self, capsys: pytest.CaptureFixture[str]) -> None:
        main(["test", "--mock"])
        captured = capsys.readouterr()
        assert "PASSED" in captured.out
