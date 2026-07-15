# Tennis Value

Tennis Value is a local, single-user Python research tool for ATP match-win
probability modeling, bookmaker-odds comparison, paper bet tracking, and
historical performance visualization.

This repository is currently bootstrapped only. It contains project structure,
developer tooling, and a minimal CLI. Data ingestion, feature engineering, model
training, backtesting, SQLite journaling, and Streamlit workflows are planned
for later tasks.

## Requirements

- Python 3.12+
- Local CSV/XLS/XLSX tennis data files will eventually live in
  `data/raw/tennis_data/`.

## Setup

```bash
python -m pip install -e ".[dev]"
```

## Developer Checks

```bash
pytest -q
ruff check .
mypy src
python -m tennis_value.cli version
```

## Product Boundary

This project is a research and decision-support tool. It does not log in to
bookmakers, place real bets, scrape authenticated pages, support live betting,
or claim guaranteed profitability.
