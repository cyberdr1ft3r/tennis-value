from __future__ import annotations

import re

import pandas as pd

from tennis_value.orientation import (
    generate_match_id,
    normalize_for_match,
    normalize_player_name,
    normalize_text,
    orient_players,
)


def test_text_normalization_handles_spacing_unicode_punctuation_and_missing() -> None:
    assert normalize_text(" Juan\u00a0 Martin   del Potro ") == "Juan Martin del Potro"
    assert normalize_text("O\u2019Connell, Christopher") == "O'Connell, Christopher"
    assert normalize_text("Semi\u2013Final") == "Semi-Final"
    assert normalize_text("\u200b Novak Djokovic ") == "Novak Djokovic"
    assert normalize_text(None) is None


def test_player_matching_normalization_is_deterministic_but_display_readable() -> None:
    first = normalize_player_name("O\u2019Connell, Christopher")
    second = normalize_player_name("O'Connell, Christopher")

    assert first is not None
    assert second is not None
    assert first.display == "O'Connell, Christopher"
    assert first.normalized == second.normalized == "o connell christopher"
    assert normalize_for_match("Juan  Martin del Potro") == "juan martin del potro"


def test_winner_can_be_player_1_or_player_2_independent_of_result() -> None:
    winner_first = orient_players("Alex Example", "Boris Sample")
    winner_second = orient_players("Zed Winner", "Adam Loser")

    assert winner_first is not None
    assert winner_first.player_1_display == "Alex Example"
    assert winner_first.player_1_won is True
    assert winner_first.swapped is False

    assert winner_second is not None
    assert winner_second.player_1_display == "Adam Loser"
    assert winner_second.player_2_display == "Zed Winner"
    assert winner_second.player_1_won is False
    assert winner_second.swapped is True


def test_match_id_is_stable_and_24_hex_characters() -> None:
    first = generate_match_id(
        pd.Timestamp("2024-01-02"),
        "brisbane international",
        "R32",
        "o connell christopher",
        "juan martin del potro",
    )
    second = generate_match_id(
        pd.Timestamp("2024-01-02"),
        "brisbane international",
        "R32",
        "o connell christopher",
        "juan martin del potro",
    )
    different = generate_match_id(
        pd.Timestamp("2024-01-03"),
        "brisbane international",
        "R32",
        "o connell christopher",
        "juan martin del potro",
    )

    assert first == second
    assert first != different
    assert re.fullmatch(r"[0-9a-f]{24}", first)


def test_match_id_ignores_source_winner_loser_order_after_orientation() -> None:
    first_orientation = orient_players("O\u2019Connell, Christopher", "Juan  Martin del Potro")
    second_orientation = orient_players("Juan Martin del Potro", "O'Connell, Christopher")

    assert first_orientation is not None
    assert second_orientation is not None
    first = generate_match_id(
        pd.Timestamp("2024-01-02"),
        "brisbane international",
        "R32",
        first_orientation.player_1_normalized,
        first_orientation.player_2_normalized,
    )
    second = generate_match_id(
        pd.Timestamp("2024-01-02"),
        "brisbane international",
        "R32",
        second_orientation.player_1_normalized,
        second_orientation.player_2_normalized,
    )

    assert first_orientation.player_1_normalized == second_orientation.player_1_normalized
    assert first_orientation.player_2_normalized == second_orientation.player_2_normalized
    assert first == second
