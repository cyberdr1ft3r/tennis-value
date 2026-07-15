from __future__ import annotations

from datetime import UTC, date, datetime

import pytest
from pydantic import ValidationError

from tennis_value.schemas import (
    CanonicalMatch,
    FeatureRow,
    PaperBet,
    ProbabilityPrediction,
    RawMatch,
    ValueAssessment,
)

MATCH_ID = "0123456789abcdef01234567"
NOW = datetime(2026, 7, 15, 12, 0, tzinfo=UTC)


def test_raw_match_accepts_minimal_valid_source_record() -> None:
    match = RawMatch(
        match_date=date(2024, 1, 2),
        tournament="Brisbane",
        surface="Hard",
        winner="Player A",
        loser="Player B",
        source_file="2024.csv",
    )

    assert match.surface == "Hard"
    assert match.winner == "Player A"


def test_raw_match_rejects_unsupported_surface_and_same_player() -> None:
    with pytest.raises(ValidationError):
        RawMatch(
            match_date=date(2024, 1, 2),
            tournament="Brisbane",
            surface="Carpet",  # type: ignore[arg-type]
            winner="Player A",
            loser="Player B",
            source_file="2024.csv",
        )

    with pytest.raises(ValidationError):
        RawMatch(
            match_date=date(2024, 1, 2),
            tournament="Brisbane",
            surface="Hard",
            winner="Player A",
            loser="player a",
            source_file="2024.csv",
        )


def test_canonical_match_contains_required_master_schema_fields() -> None:
    match = CanonicalMatch(
        match_id=MATCH_ID,
        match_date=date(2024, 1, 2),
        tournament="Brisbane",
        surface="Hard",
        round="R32",
        best_of=3,
        player_1="Player A",
        player_2="Player B",
        player_1_rank=10,
        player_2_rank=20,
        player_1_odds=1.8,
        player_2_odds=2.1,
        player_1_won=True,
        is_retirement=False,
        source_file="2024.csv",
    )

    assert match.match_id == MATCH_ID
    assert match.player_1_won is True


def test_canonical_match_allows_missing_optional_rankings_and_odds() -> None:
    match = CanonicalMatch(
        match_id=MATCH_ID,
        match_date=date(2024, 1, 2),
        tournament="Brisbane",
        surface="Hard",
        round="R32",
        best_of=3,
        player_1="Player A",
        player_2="Player B",
        player_1_won=True,
        source_file="2024.csv",
    )

    assert match.player_1_rank is None
    assert match.player_2_odds is None


def test_canonical_match_rejects_invalid_match_id_and_odds() -> None:
    with pytest.raises(ValidationError):
        CanonicalMatch(
            match_id="not-a-match-id",
            match_date=date(2024, 1, 2),
            tournament="Brisbane",
            surface="Hard",
            round="R32",
            best_of=3,
            player_1="Player A",
            player_2="Player B",
            player_1_odds=1.0,
            player_1_won=True,
            source_file="2024.csv",
        )


def test_match_schemas_reject_invalid_best_of_values() -> None:
    with pytest.raises(ValidationError):
        RawMatch(
            match_date=date(2024, 1, 2),
            tournament="Brisbane",
            surface="Hard",
            best_of=4,  # type: ignore[arg-type]
            winner="Player A",
            loser="Player B",
            source_file="2024.csv",
        )

    with pytest.raises(ValidationError):
        CanonicalMatch(
            match_id=MATCH_ID,
            match_date=date(2024, 1, 2),
            tournament="Brisbane",
            surface="Hard",
            round="R32",
            best_of=1,  # type: ignore[arg-type]
            player_1="Player A",
            player_2="Player B",
            player_1_won=True,
            source_file="2024.csv",
        )


def test_feature_row_accepts_pre_match_feature_fields() -> None:
    row = FeatureRow(
        match_id=MATCH_ID,
        match_date=date(2024, 1, 2),
        surface="Clay",
        player_1_won=False,
        overall_elo_diff=12.5,
        surface_elo_diff=-4.0,
        matches_last_14d_diff=1,
        history_count_min=3,
        best_of_5=False,
        surface_clay=True,
        surface_grass=False,
    )

    assert row.surface_clay is True
    assert row.history_count_min == 3


def test_probability_prediction_requires_complementary_probabilities() -> None:
    prediction = ProbabilityPrediction(
        prediction_id="pred-1",
        match_id=MATCH_ID,
        model_version="model_v1",
        predicted_at=NOW,
        player_1_win_probability=0.6,
        player_2_win_probability=0.4,
    )

    assert prediction.player_1_win_probability == 0.6

    with pytest.raises(ValidationError):
        ProbabilityPrediction(
            prediction_id="pred-2",
            match_id=MATCH_ID,
            model_version="model_v1",
            predicted_at=NOW,
            player_1_win_probability=0.6,
            player_2_win_probability=0.5,
        )


def test_value_assessment_validates_eligible_selection_details() -> None:
    assessment = ValueAssessment(
        match_id=MATCH_ID,
        selection="player_1",
        eligible=True,
        model_probability=0.6,
        market_probability=0.54,
        edge=0.06,
        expected_value=0.08,
        decimal_odds=1.8,
    )

    assert assessment.eligible is True

    with pytest.raises(ValidationError):
        ValueAssessment(match_id=MATCH_ID, selection="none", eligible=True)


def test_paper_bet_requires_consistent_settlement_state() -> None:
    open_bet = PaperBet(
        bet_id="bet-1",
        match_id=MATCH_ID,
        selection="player_1",
        placed_at=NOW,
        stake=50,
        decimal_odds=2.0,
        model_probability=0.58,
        edge=0.05,
        expected_value=0.16,
    )

    assert open_bet.status == "open"

    with pytest.raises(ValidationError):
        PaperBet(
            bet_id="bet-2",
            match_id=MATCH_ID,
            selection="player_1",
            placed_at=NOW,
            stake=50,
            decimal_odds=2.0,
            model_probability=0.58,
            edge=0.05,
            expected_value=0.16,
            status="won",
        )

    settled_bet = PaperBet(
        bet_id="bet-3",
        match_id=MATCH_ID,
        selection="player_2",
        placed_at=NOW,
        stake=50,
        decimal_odds=2.0,
        model_probability=0.58,
        edge=0.05,
        expected_value=0.16,
        status="lost",
        profit_loss=-50,
    )

    assert settled_bet.profit_loss == -50
