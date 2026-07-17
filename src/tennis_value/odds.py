"""Decimal odds validation and no-vig probability conversion."""

from __future__ import annotations

import math
from typing import Any

from pydantic import BaseModel, ConfigDict

PROBABILITY_SUM_TOLERANCE = 1e-9


class NoVigMarket(BaseModel):
    """Two-way market probabilities after proportional margin removal."""

    model_config = ConfigDict(frozen=True)

    player_1_odds: float
    player_2_odds: float
    raw_player_1_probability: float
    raw_player_2_probability: float
    overround: float
    player_1_market_probability: float
    player_2_market_probability: float


def validate_decimal_odds(value: Any) -> float:
    """Return a valid decimal price or raise ValueError."""
    try:
        decimal_odds = float(value)
    except (TypeError, ValueError) as exc:
        msg = "decimal odds must be numeric"
        raise ValueError(msg) from exc
    if not math.isfinite(decimal_odds):
        msg = "decimal odds must be finite"
        raise ValueError(msg)
    if decimal_odds <= 1.0:
        msg = "decimal odds must be greater than 1.0"
        raise ValueError(msg)
    return decimal_odds


def implied_probability(decimal_odds: float) -> float:
    """Convert valid decimal odds to raw implied probability."""
    odds = validate_decimal_odds(decimal_odds)
    return 1.0 / odds


def remove_two_way_margin(player_1_odds: Any, player_2_odds: Any) -> NoVigMarket:
    """Return no-vig probabilities for a valid two-player market."""
    odds_1 = validate_decimal_odds(player_1_odds)
    odds_2 = validate_decimal_odds(player_2_odds)
    raw_1 = implied_probability(odds_1)
    raw_2 = implied_probability(odds_2)
    overround = raw_1 + raw_2
    if not math.isfinite(overround) or overround <= 0:
        msg = "market overround must be finite and greater than zero"
        raise ValueError(msg)
    market_1 = raw_1 / overround
    market_2 = raw_2 / overround
    if not math.isclose(market_1 + market_2, 1.0, abs_tol=PROBABILITY_SUM_TOLERANCE):
        msg = "no-vig probabilities must sum to 1"
        raise ValueError(msg)
    return NoVigMarket(
        player_1_odds=odds_1,
        player_2_odds=odds_2,
        raw_player_1_probability=raw_1,
        raw_player_2_probability=raw_2,
        overround=overround,
        player_1_market_probability=market_1,
        player_2_market_probability=market_2,
    )


__all__ = [
    "NoVigMarket",
    "PROBABILITY_SUM_TOLERANCE",
    "implied_probability",
    "remove_two_way_margin",
    "validate_decimal_odds",
]
