"""Configuration models for Tennis Value."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SupportedSurface = Literal["Hard", "Clay", "Grass"]


class FrozenModel(BaseModel):
    """Base model for immutable configuration records."""

    model_config = ConfigDict(frozen=True)


class PipelinePaths(FrozenModel):
    """Project-root-relative paths used by the local pipeline."""

    project_root: Path = Path(".")
    raw_data_dir: Path = Path("data/raw")
    tennis_data_dir: Path = Path("data/raw/tennis_data")
    processed_data_dir: Path = Path("data/processed")
    sample_data_dir: Path = Path("data/samples")
    models_dir: Path = Path("models")
    reports_dir: Path = Path("reports")
    state_dir: Path = Path("state")
    sqlite_path: Path = Path("state/tennis_value.sqlite3")

    @field_validator(
        "raw_data_dir",
        "tennis_data_dir",
        "processed_data_dir",
        "sample_data_dir",
        "models_dir",
        "reports_dir",
        "state_dir",
        "sqlite_path",
    )
    @classmethod
    def _pipeline_paths_must_be_relative(cls, value: Path) -> Path:
        if value.is_absolute():
            msg = "pipeline paths must be relative to project_root"
            raise ValueError(msg)
        return value

    def resolve(self, path: Path) -> Path:
        """Resolve a configured relative path against the project root."""
        if path.is_absolute():
            msg = "expected a path relative to project_root"
            raise ValueError(msg)
        return self.project_root / path

    @property
    def raw_data_path(self) -> Path:
        return self.resolve(self.raw_data_dir)

    @property
    def tennis_data_path(self) -> Path:
        return self.resolve(self.tennis_data_dir)

    @property
    def processed_data_path(self) -> Path:
        return self.resolve(self.processed_data_dir)

    @property
    def sample_data_path(self) -> Path:
        return self.resolve(self.sample_data_dir)

    @property
    def models_path(self) -> Path:
        return self.resolve(self.models_dir)

    @property
    def reports_path(self) -> Path:
        return self.resolve(self.reports_dir)

    @property
    def state_path(self) -> Path:
        return self.resolve(self.state_dir)

    @property
    def database_path(self) -> Path:
        return self.resolve(self.sqlite_path)


class EloConfig(FrozenModel):
    """Configurable Elo parameters."""

    initial_rating: float = Field(default=1500.0, gt=0)
    k_factor: float = Field(default=32.0, gt=0)
    elo_scale: float = Field(default=400.0, gt=0)


class DateSplitConfig(FrozenModel):
    """Chronological season and model split boundaries."""

    seasons: tuple[int, ...] = tuple(range(2020, 2026))
    train_start: date = date(2020, 1, 1)
    train_end: date = date(2023, 12, 31)
    validation_start: date = date(2024, 1, 1)
    validation_end: date = date(2024, 12, 31)
    test_start: date = date(2025, 1, 1)
    test_end: date = date(2025, 12, 31)

    @field_validator("seasons")
    @classmethod
    def _seasons_must_be_unique_and_ordered(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            msg = "at least one season is required"
            raise ValueError(msg)
        if tuple(sorted(value)) != value or len(set(value)) != len(value):
            msg = "seasons must be unique and sorted"
            raise ValueError(msg)
        return value

    @model_validator(mode="after")
    def _date_ranges_must_be_chronological(self) -> Self:
        ranges = (
            ("train", self.train_start, self.train_end),
            ("validation", self.validation_start, self.validation_end),
            ("test", self.test_start, self.test_end),
        )
        for name, start, end in ranges:
            if start > end:
                msg = f"{name} date range is reversed"
                raise ValueError(msg)
        if self.train_end >= self.validation_start:
            msg = "train and validation date ranges overlap"
            raise ValueError(msg)
        if self.validation_end >= self.test_start:
            msg = "validation and test date ranges overlap"
            raise ValueError(msg)
        return self


class ValueThresholds(FrozenModel):
    """Default research thresholds for value detection."""

    min_model_probability: float = Field(default=0.55, ge=0, le=1)
    min_edge: float = Field(default=0.04, ge=-1, le=1)
    min_expected_value: float = Field(default=0.03, ge=-1)
    min_odds: float = Field(default=1.50, gt=1)
    max_odds: float = Field(default=3.50, gt=1)

    @model_validator(mode="after")
    def _odds_range_must_be_ordered(self) -> Self:
        if self.min_odds > self.max_odds:
            msg = "min_odds must be less than or equal to max_odds"
            raise ValueError(msg)
        return self


class BacktestConfig(FrozenModel):
    """Configuration for flat-stake paper backtests."""

    starting_bankroll: float = Field(default=10000.0, gt=0)
    flat_stake_fraction: float = Field(default=0.005, gt=0, le=1)
    void_retirements: bool = True


class AppConfig(FrozenModel):
    """Top-level application configuration."""

    paths: PipelinePaths = Field(default_factory=PipelinePaths)
    elo: EloConfig = Field(default_factory=EloConfig)
    date_splits: DateSplitConfig = Field(default_factory=DateSplitConfig)
    value_thresholds: ValueThresholds = Field(default_factory=ValueThresholds)
    backtest: BacktestConfig = Field(default_factory=BacktestConfig)
    supported_surfaces: tuple[SupportedSurface, ...] = ("Hard", "Clay", "Grass")

    @field_validator("supported_surfaces")
    @classmethod
    def _supported_surfaces_must_be_known(
        cls, value: tuple[SupportedSurface, ...]
    ) -> tuple[SupportedSurface, ...]:
        if not value:
            msg = "at least one supported surface is required"
            raise ValueError(msg)
        return value


__all__ = [
    "AppConfig",
    "BacktestConfig",
    "DateSplitConfig",
    "EloConfig",
    "PipelinePaths",
    "SupportedSurface",
    "ValueThresholds",
]
