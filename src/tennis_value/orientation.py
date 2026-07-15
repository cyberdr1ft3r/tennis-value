"""Neutral player orientation and stable match identifiers."""

from __future__ import annotations

import hashlib
import re
import unicodedata
from dataclasses import dataclass
from datetime import date
from typing import Any

import pandas as pd

INVISIBLE_TRANSLATION = dict.fromkeys(
    ord(char) for char in "\u200b\u200c\u200d\ufeff\u2060"
)
APOSTROPHE_TRANSLATION = str.maketrans(
    {
        "\u2018": "'",
        "\u2019": "'",
        "\u201a": "'",
        "\u201b": "'",
        "\u2032": "'",
        "\u02bc": "'",
    }
)
QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u201e": '"',
        "\u201f": '"',
        "\u00ab": '"',
        "\u00bb": '"',
    }
)
DASH_TRANSLATION = str.maketrans(
    {
        "\u2010": "-",
        "\u2011": "-",
        "\u2012": "-",
        "\u2013": "-",
        "\u2014": "-",
        "\u2212": "-",
    }
)


@dataclass(frozen=True)
class NormalizedName:
    """Display and comparison forms for a player name."""

    display: str
    normalized: str


@dataclass(frozen=True)
class OrientedPlayers:
    """Neutral player orientation for a source winner and loser."""

    player_1_display: str
    player_2_display: str
    player_1_normalized: str
    player_2_normalized: str
    player_1_won: bool
    swapped: bool


def normalize_text(value: Any) -> str | None:
    """Normalize readable text without changing its intent."""
    if value is None or value is pd.NA:
        return None
    try:
        if pd.isna(value):
            return None
    except TypeError:
        pass

    text = str(value)
    text = text.translate(INVISIBLE_TRANSLATION)
    text = text.replace("\u00a0", " ")
    text = text.translate(APOSTROPHE_TRANSLATION)
    text = text.translate(QUOTE_TRANSLATION)
    text = text.translate(DASH_TRANSLATION)
    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_for_match(value: Any) -> str | None:
    """Normalize text for deterministic comparison and IDs."""
    text = normalize_text(value)
    if text is None:
        return None
    text = text.casefold()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^\w\s]", " ", text, flags=re.ASCII)
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or None


def normalize_player_name(value: Any) -> NormalizedName | None:
    """Return display and matching forms for a player name."""
    display = normalize_text(value)
    normalized = normalize_for_match(value)
    if display is None or normalized is None:
        return None
    return NormalizedName(display=display, normalized=normalized)


def orient_players(winner: Any, loser: Any) -> OrientedPlayers | None:
    """Orient players lexicographically by normalized name, independent of result."""
    winner_name = normalize_player_name(winner)
    loser_name = normalize_player_name(loser)
    if winner_name is None or loser_name is None:
        return None
    if winner_name.normalized == loser_name.normalized:
        return None
    if winner_name.normalized <= loser_name.normalized:
        return OrientedPlayers(
            player_1_display=winner_name.display,
            player_2_display=loser_name.display,
            player_1_normalized=winner_name.normalized,
            player_2_normalized=loser_name.normalized,
            player_1_won=True,
            swapped=False,
        )
    return OrientedPlayers(
        player_1_display=loser_name.display,
        player_2_display=winner_name.display,
        player_1_normalized=loser_name.normalized,
        player_2_normalized=winner_name.normalized,
        player_1_won=False,
        swapped=True,
    )


def generate_match_id(
    match_date: date | pd.Timestamp,
    tournament_normalized: str,
    round_normalized: str,
    player_1_normalized: str,
    player_2_normalized: str,
) -> str:
    """Generate a stable 24-character SHA-256-derived match ID."""
    date_text = pd.Timestamp(match_date).strftime("%Y-%m-%d")
    canonical = "|".join(
        [
            date_text,
            tournament_normalized,
            round_normalized,
            player_1_normalized,
            player_2_normalized,
        ]
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:24]


__all__ = [
    "NormalizedName",
    "OrientedPlayers",
    "generate_match_id",
    "normalize_for_match",
    "normalize_player_name",
    "normalize_text",
    "orient_players",
]
