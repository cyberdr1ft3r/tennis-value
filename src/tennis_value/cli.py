"""Command-line interface for Tennis Value."""

from __future__ import annotations

import typer
from rich.console import Console

from tennis_value import __version__

app = typer.Typer(help="Tennis Value research tooling.")
console = Console()


@app.callback()
def main() -> None:
    """Tennis Value command group."""


@app.command()
def version() -> None:
    """Print the installed Tennis Value version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
