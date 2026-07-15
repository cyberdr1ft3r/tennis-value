"""Typed domain schemas for Tennis Value."""

from __future__ import annotations

from datetime import date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tennis_value.config import SupportedSurface

PlayerSelection = Literal["player_1", "player_2"]
ValueSelection = Literal["player_1", "player_2", "none"]
BetStatus = Literal["open", "won", "lost", "void"]
BestOf = Literal[3, 5]


class DomainModel(BaseModel):
    """Base model for domain records."""

    model_config = ConfigDict(frozen=True, str_strip_whitespace=True)


def _non_empty(value: str) -> str:
    if not value:
        msg = "value must not be empty"
        raise ValueError(msg)
    return value


class RawMatch(DomainModel):
    """A source-row-level match record before canonical orientation."""

    match_date: date
    tournament: str
    surface: SupportedSurface
    round: str | None = None
    best_of: BestOf | None = None
    winner: str
    loser: str
    winner_rank: int | None = Field(default=None, gt=0)
    loser_rank: int | None = Field(default=None, gt=0)
    winner_odds: float | None = Field(default=None, gt=1)
    loser_odds: float | None = Field(default=None, gt=1)
    status: str | None = None
    source_file: str

    @field_validator("tournament", "winner", "loser", "source_file")
    @classmethod
    def _required_strings_must_not_be_empty(cls, value: str) -> str:
        return _non_empty(value)

    @model_validator(mode="after")
    def _players_must_differ(self) -> Self:
        if self.winner.casefold() == self.loser.casefold():
            msg = "winner and loser must be different players"
            raise ValueError(msg)
        return self


class CanonicalMatch(DomainModel):
    """Neutral, result-independent match orientation."""

    match_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    match_date: date
    tournament: str
    surface: SupportedSurface
    round: str
    best_of: BestOf
    player_1: str
    player_2: str
    player_1_rank: int | None = Field(default=None, gt=0)
    player_2_rank: int | None = Field(default=None, gt=0)
    player_1_odds: float | None = Field(default=None, gt=1)
    player_2_odds: float | None = Field(default=None, gt=1)
    player_1_won: bool
    is_retirement: bool = False
    source_file: str

    @field_validator("tournament", "round", "player_1", "player_2", "source_file")
    @classmethod
    def _required_strings_must_not_be_empty(cls, value: str) -> str:
        return _non_empty(value)

    @model_validator(mode="after")
    def _players_must_differ(self) -> Self:
        if self.player_1.casefold() == self.player_2.casefold():
            msg = "player_1 and player_2 must be different players"
            raise ValueError(msg)
        return self


class FeatureRow(DomainModel):
    """Pre-match model feature record."""

    match_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    match_date: date
    surface: SupportedSurface
    player_1_won: bool
    overall_elo_diff: float
    surface_elo_diff: float
    log_rank_diff: float | None = None
    recent_10_win_rate_diff: float | None = None
    surface_recent_10_win_rate_diff: float | None = None
    days_since_last_match_diff: float | None = None
    matches_last_14d_diff: int
    history_count_min: int = Field(ge=0)
    best_of_5: bool
    surface_clay: bool
    surface_grass: bool


class ProbabilityPrediction(DomainModel):
    """A model probability prediction for one canonical match."""

    prediction_id: str
    match_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    model_version: str
    predicted_at: datetime
    player_1_win_probability: float = Field(ge=0, le=1)
    player_2_win_probability: float = Field(ge=0, le=1)

    @field_validator("prediction_id", "model_version")
    @classmethod
    def _required_strings_must_not_be_empty(cls, value: str) -> str:
        return _non_empty(value)

    @model_validator(mode="after")
    def _probabilities_must_sum_to_one(self) -> Self:
        if abs((self.player_1_win_probability + self.player_2_win_probability) - 1.0) > 1e-9:
            msg = "player win probabilities must sum to 1"
            raise ValueError(msg)
        return self


class ValueAssessment(DomainModel):
    """A threshold-based value assessment for one match."""

    match_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    selection: ValueSelection
    eligible: bool
    model_probability: float | None = Field(default=None, ge=0, le=1)
    market_probability: float | None = Field(default=None, ge=0, le=1)
    edge: float | None = None
    expected_value: float | None = None
    decimal_odds: float | None = Field(default=None, gt=1)
    skip_reasons: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _eligible_assessment_must_have_metrics(self) -> Self:
        if self.eligible and self.selection == "none":
            msg = "eligible assessments must select a player"
            raise ValueError(msg)
        if self.eligible:
            required_values = (
                self.model_probability,
                self.market_probability,
                self.edge,
                self.expected_value,
                self.decimal_odds,
            )
            if any(value is None for value in required_values):
                msg = "eligible assessments require probability, edge, EV, and odds"
                raise ValueError(msg)
        return self


class PaperBet(DomainModel):
    """A logged paper bet or simulated backtest selection."""

    bet_id: str
    match_id: str = Field(pattern=r"^[0-9a-f]{24}$")
    selection: PlayerSelection
    placed_at: datetime
    stake: float = Field(gt=0)
    decimal_odds: float = Field(gt=1)
    model_probability: float = Field(ge=0, le=1)
    edge: float
    expected_value: float
    status: BetStatus = "open"
    profit_loss: float | None = None

    @field_validator("bet_id")
    @classmethod
    def _required_strings_must_not_be_empty(cls, value: str) -> str:
        return _non_empty(value)

    @model_validator(mode="after")
    def _open_bets_are_unsettled(self) -> Self:
        if self.status == "open" and self.profit_loss is not None:
            msg = "open bets must not have profit_loss"
            raise ValueError(msg)
        if self.status != "open" and self.profit_loss is None:
            msg = "settled bets require profit_loss"
            raise ValueError(msg)
        return self


__all__ = [
    "BetStatus",
    "BestOf",
    "CanonicalMatch",
    "FeatureRow",
    "PaperBet",
    "PlayerSelection",
    "ProbabilityPrediction",
    "RawMatch",
    "ValueAssessment",
    "ValueSelection",
]
