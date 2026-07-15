from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from tennis_value.config import (
    AppConfig,
    BacktestConfig,
    DateSplitConfig,
    EloConfig,
    PipelinePaths,
    ValueThresholds,
)


def test_app_config_defaults_match_task_two() -> None:
    config = AppConfig()

    assert config.date_splits.seasons == (2020, 2021, 2022, 2023, 2024, 2025)
    assert config.date_splits.train_end == date(2023, 12, 31)
    assert config.date_splits.validation_start == date(2024, 1, 1)
    assert config.date_splits.validation_end == date(2024, 12, 31)
    assert config.date_splits.test_start == date(2025, 1, 1)
    assert config.date_splits.test_end == date(2025, 12, 31)
    assert config.elo.initial_rating == 1500
    assert config.elo.k_factor == 32
    assert config.elo.elo_scale == 400
    assert config.backtest.starting_bankroll == 10000
    assert config.backtest.flat_stake_fraction == 0.005
    assert config.value_thresholds.min_model_probability == 0.55
    assert config.value_thresholds.min_edge == 0.04
    assert config.value_thresholds.min_expected_value == 0.03
    assert config.value_thresholds.min_odds == 1.50
    assert config.value_thresholds.max_odds == 3.50
    assert config.supported_surfaces == ("Hard", "Clay", "Grass")


def test_paths_resolve_relative_to_configurable_project_root() -> None:
    paths = PipelinePaths(project_root=Path("project"))

    assert paths.tennis_data_path == Path("project/data/raw/tennis_data")
    assert paths.database_path == Path("project/state/tennis_value.sqlite3")


def test_pipeline_subpaths_must_be_relative() -> None:
    with pytest.raises(ValidationError):
        PipelinePaths(models_dir=Path("C:/absolute/models"))


def test_date_splits_reject_reversed_or_overlapping_ranges() -> None:
    with pytest.raises(ValidationError):
        DateSplitConfig(train_start=date(2024, 1, 1), train_end=date(2023, 12, 31))

    with pytest.raises(ValidationError):
        DateSplitConfig(validation_start=date(2023, 12, 31))

    with pytest.raises(ValidationError):
        DateSplitConfig(test_start=date(2024, 12, 31))


def test_date_splits_reject_unsorted_or_duplicate_seasons() -> None:
    with pytest.raises(ValidationError):
        DateSplitConfig(seasons=(2020, 2022, 2021))

    with pytest.raises(ValidationError):
        DateSplitConfig(seasons=(2020, 2020))


def test_invalid_elo_parameters_are_rejected() -> None:
    with pytest.raises(ValidationError):
        EloConfig(initial_rating=0)
    with pytest.raises(ValidationError):
        EloConfig(k_factor=-1)
    with pytest.raises(ValidationError):
        EloConfig(elo_scale=0)


def test_invalid_value_thresholds_are_rejected() -> None:
    with pytest.raises(ValidationError):
        ValueThresholds(min_model_probability=1.01)
    with pytest.raises(ValidationError):
        ValueThresholds(min_odds=3.5, max_odds=1.5)
    with pytest.raises(ValidationError):
        ValueThresholds(min_odds=1.0)


def test_invalid_backtest_values_are_rejected() -> None:
    with pytest.raises(ValidationError):
        BacktestConfig(starting_bankroll=0)
    with pytest.raises(ValidationError):
        BacktestConfig(flat_stake_fraction=0)
    with pytest.raises(ValidationError):
        BacktestConfig(flat_stake_fraction=1.01)


def test_unsupported_surfaces_are_rejected() -> None:
    with pytest.raises(ValidationError):
        AppConfig(supported_surfaces=("Hard", "Indoor"))  # type: ignore[arg-type]


def test_configuration_models_are_immutable() -> None:
    config = AppConfig()

    with pytest.raises(ValidationError):
        config.backtest.starting_bankroll = 1
