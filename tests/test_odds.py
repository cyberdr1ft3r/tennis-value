from __future__ import annotations

import math

import pytest

from tennis_value.odds import (
    implied_probability,
    remove_two_way_margin,
    validate_decimal_odds,
)


@pytest.mark.parametrize("value", [1.01, 1.5, 2, "2.25"])
def test_validate_decimal_odds_accepts_numeric_prices_above_one(value: object) -> None:
    assert validate_decimal_odds(value) == pytest.approx(float(value))


@pytest.mark.parametrize("value", [None, "abc", 1.0, 0, -2, math.nan, math.inf, -math.inf])
def test_validate_decimal_odds_rejects_invalid_prices(value: object) -> None:
    with pytest.raises(ValueError):
        validate_decimal_odds(value)


def test_implied_probability_converts_decimal_odds() -> None:
    assert implied_probability(2.0) == pytest.approx(0.5)
    assert implied_probability(4.0) == pytest.approx(0.25)


def test_remove_two_way_margin_normalizes_market_probabilities() -> None:
    market = remove_two_way_margin(1.8, 2.1)

    assert market.player_1_odds == pytest.approx(1.8)
    assert market.player_2_odds == pytest.approx(2.1)
    assert market.raw_player_1_probability == pytest.approx(1 / 1.8)
    assert market.raw_player_2_probability == pytest.approx(1 / 2.1)
    assert market.overround == pytest.approx((1 / 1.8) + (1 / 2.1))
    assert market.player_1_market_probability + market.player_2_market_probability == pytest.approx(
        1.0
    )


def test_remove_two_way_margin_handles_even_market() -> None:
    market = remove_two_way_margin(2.0, 2.0)

    assert market.overround == pytest.approx(1.0)
    assert market.player_1_market_probability == pytest.approx(0.5)
    assert market.player_2_market_probability == pytest.approx(0.5)


@pytest.mark.parametrize("player_1_odds, player_2_odds", [(1.0, 2.0), (2.0, None), (2.0, "bad")])
def test_remove_two_way_margin_rejects_invalid_markets(
    player_1_odds: object,
    player_2_odds: object,
) -> None:
    with pytest.raises(ValueError):
        remove_two_way_margin(player_1_odds, player_2_odds)
