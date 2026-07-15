from __future__ import annotations

from typer.testing import CliRunner

import tennis_value
from tennis_value.cli import app


def test_package_imports() -> None:
    assert tennis_value.__version__ == "0.1.0"


def test_cli_version() -> None:
    result = CliRunner().invoke(app, ["version"])

    assert result.exit_code == 0
    assert "0.1.0" in result.output
