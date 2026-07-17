from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

from tennis_value.config import ValueThresholds
from tennis_value.value import (
    REASON_ORDER,
    SelectionAssessment,
    assess_match_value,
    assess_prediction_dataframe,
    assess_predictions_with_summary,
    assess_selection,
    calculate_edge,
    calculate_expected_value,
    validate_model_probability,
    validate_probability_pair,
    write_value_outputs,
)


def _prediction_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "match_id": "m01",
        "match_date": pd.Timestamp("2025-01-01"),
        "partition": "test",
        "surface": "Hard",
        "player_1": "Alice",
        "player_2": "Bob",
        "actual_player_1_won": True,
        "predicted_player_1_probability": 0.62,
        "predicted_player_2_probability": 0.38,
        "player_1_odds": 2.0,
        "player_2_odds": 1.8,
        "model_version": "model_v1",
    }
    row.update(overrides)
    return row


def test_edge_and_expected_value_formulas() -> None:
    assert calculate_edge(0.6, 0.55) == pytest.approx(0.05)
    assert calculate_expected_value(0.6, 2.0) == pytest.approx(0.2)
    assert calculate_expected_value(0.5, 2.0) == pytest.approx(0.0)
    assert calculate_expected_value(0.4, 2.0) == pytest.approx(-0.2)


@pytest.mark.parametrize("value", [0.0, 0.5, 1.0])
def test_model_probability_accepts_valid_range(value: float) -> None:
    assert validate_model_probability(value) == pytest.approx(value)


@pytest.mark.parametrize("value", [-0.01, 1.01, "bad", None])
def test_model_probability_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError):
        validate_model_probability(value)


def test_probability_pair_must_sum_to_one() -> None:
    assert validate_probability_pair(0.55, 0.45) == pytest.approx((0.55, 0.45))
    with pytest.raises(ValueError):
        validate_probability_pair(0.55, 0.5)


def test_thresholds_are_inclusive_at_exact_boundaries() -> None:
    assessment = assess_selection(
        match_id="m01",
        selection_side="player_1",
        selection_player="Alice",
        model_probability=0.55,
        market_probability=0.51,
        raw_implied_probability=0.5,
        decimal_odds=1.8727272727272728,
        overround=1.0,
        thresholds=ValueThresholds(),
    )

    assert assessment.eligible is True
    assert assessment.reason_codes == ()
    assert assessment.edge == pytest.approx(0.04)
    assert assessment.expected_value == pytest.approx(0.03)


@pytest.mark.parametrize(
    ("kwargs", "reason"),
    [
        ({"model_probability": 0.549}, "probability_below_threshold"),
        ({"market_probability": 0.581}, "edge_below_threshold"),
        ({"decimal_odds": 1.6}, "expected_value_below_threshold"),
        ({"decimal_odds": 1.49}, "odds_below_minimum"),
        ({"decimal_odds": 3.51}, "odds_above_maximum"),
    ],
)
def test_selection_records_failed_threshold_reasons(
    kwargs: dict[str, float],
    reason: str,
) -> None:
    values = {
        "model_probability": 0.58,
        "market_probability": 0.50,
        "raw_implied_probability": 0.5,
        "decimal_odds": 2.0,
        "overround": 1.0,
    }
    values.update(kwargs)

    assessment = assess_selection(
        match_id="m01",
        selection_side="player_1",
        selection_player="Alice",
        thresholds=ValueThresholds(),
        **values,
    )

    assert assessment.eligible is False
    assert reason in assessment.reason_codes


def test_valid_match_recommends_one_player() -> None:
    assessment = assess_match_value(_prediction_row())

    assert assessment.has_recommendation is True
    assert assessment.recommended_side == "player_1"
    assert assessment.recommended_player == "Alice"
    assert assessment.player_1_assessment.edge is not None
    assert assessment.player_1_assessment.expected_value is not None
    assert assessment.player_2_assessment.eligible is False


@pytest.mark.parametrize(
    ("overrides", "reason"),
    [
        ({"match_id": None}, "missing_match_id"),
        ({"player_2": "Alice"}, "same_player"),
        ({"surface": "Carpet"}, "unsupported_surface"),
        ({"odds_source": "inconsistent"}, "unsupported_match"),
        ({"is_retirement": True}, "retirement"),
        ({"is_walkover": True}, "walkover"),
        ({"player_1_odds": None}, "missing_odds"),
        ({"player_1_odds": 1.0}, "invalid_odds"),
        (
            {"predicted_player_1_probability": 1.1, "predicted_player_2_probability": -0.1},
            "invalid_probability",
        ),
    ],
)
def test_match_level_skip_reasons(overrides: dict[str, object], reason: str) -> None:
    assessment = assess_match_value(_prediction_row(**overrides))

    assert assessment.has_recommendation is False
    assert reason in assessment.decision_reason_codes


def test_supported_surfaces_are_accepted() -> None:
    for surface in ("Hard", "Clay", "Grass"):
        assert assess_match_value(_prediction_row(surface=surface)).surface == surface


def test_only_one_selection_is_recommended_when_both_are_eligible() -> None:
    thresholds = ValueThresholds(
        min_model_probability=0.0,
        min_edge=0.0,
        min_expected_value=0.0,
        min_odds=1.01,
        max_odds=10.0,
    )
    assessment = assess_match_value(
        _prediction_row(
            predicted_player_1_probability=0.5,
            predicted_player_2_probability=0.5,
            player_1_odds=2.0,
            player_2_odds=2.0,
        ),
        thresholds,
    )

    assert assessment.has_recommendation is True
    assert assessment.recommended_side == "player_1"
    assert "multiple_eligible_selections" in assessment.decision_reason_codes


def test_player_two_can_be_recommended() -> None:
    assessment = assess_match_value(
        _prediction_row(
            predicted_player_1_probability=0.35,
            predicted_player_2_probability=0.65,
            player_1_odds=1.7,
            player_2_odds=2.2,
        )
    )

    assert assessment.has_recommendation is True
    assert assessment.recommended_side == "player_2"
    assert assessment.recommended_player == "Bob"


def test_dataframe_assessment_is_deterministic_and_does_not_mutate_input() -> None:
    frame = pd.DataFrame(
        [
            _prediction_row(match_id="m02", match_date=pd.Timestamp("2025-01-02")),
            _prediction_row(match_id="m01", match_date=pd.Timestamp("2025-01-01")),
        ]
    )
    original = frame.copy(deep=True)

    first = assess_prediction_dataframe(frame)
    second = assess_prediction_dataframe(frame)

    pd.testing.assert_frame_equal(frame, original)
    pd.testing.assert_frame_equal(first, second)
    assert first["match_id"].tolist() == ["m01", "m02"]
    assert first["has_recommendation"].tolist() == [True, True]


def test_dataframe_rejects_missing_columns_duplicate_ids_and_bad_probabilities() -> None:
    frame = pd.DataFrame([_prediction_row()])

    with pytest.raises(ValueError, match="missing required"):
        assess_prediction_dataframe(frame.drop(columns=["model_version"]))
    with pytest.raises(ValueError, match="duplicate"):
        assess_prediction_dataframe(pd.concat([frame, frame], ignore_index=True))
    with pytest.raises(ValueError, match="invalid probability pair"):
        assess_prediction_dataframe(
            pd.DataFrame(
                [
                    _prediction_row(
                        predicted_player_1_probability=0.6,
                        predicted_player_2_probability=0.6,
                    )
                ]
            )
        )


def test_dataframe_keeps_invalid_odds_as_skipped_rows() -> None:
    result = assess_prediction_dataframe(pd.DataFrame([_prediction_row(player_1_odds=1.0)]))

    assert len(result) == 1
    assert not bool(result.loc[0, "has_recommendation"])
    assert not bool(result.loc[0, "odds_valid"])
    assert "invalid_odds" in result.loc[0, "decision_reason_codes"]


def test_summary_counts_recommendations_and_skip_reasons() -> None:
    result = assess_predictions_with_summary(
        pd.DataFrame(
            [
                _prediction_row(match_id="m01"),
                _prediction_row(match_id="m02", surface="Carpet"),
                _prediction_row(match_id="m03", player_1_odds=None),
            ]
        )
    )

    assert result.summary.rows_received == 3
    assert result.summary.rows_assessed == 3
    assert result.summary.rows_with_recommendations == 1
    assert result.summary.recommendations_player_1 == 1
    assert result.summary.recommendations_player_2 == 0
    assert result.summary.skip_reason_counts["unsupported_surface"] >= 1
    assert result.summary.skip_reason_counts["missing_odds"] >= 1
    assert result.summary.thresholds["minimum_model_probability"] == pytest.approx(0.55)


def test_write_value_outputs_creates_parquet_and_json_artifacts() -> None:
    output_dir = Path(".tmp-task9-artifacts")
    output_path = output_dir / "value.parquet"
    summary_path = output_dir / "value.json"
    output_path.unlink(missing_ok=True)
    summary_path.unlink(missing_ok=True)

    result = assess_predictions_with_summary(pd.DataFrame([_prediction_row()]))
    write_value_outputs(result, output_path, summary_path)

    written = pd.read_parquet(output_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    assert written["match_id"].tolist() == ["m01"]
    assert summary["rows_with_recommendations"] == 1


def test_reason_order_contains_expected_task9_codes() -> None:
    assert set(REASON_ORDER) >= {
        "missing_match_id",
        "same_player",
        "unsupported_surface",
        "unsupported_match",
        "retirement",
        "walkover",
        "missing_odds",
        "invalid_odds",
        "invalid_probability",
        "probability_below_threshold",
        "edge_below_threshold",
        "expected_value_below_threshold",
        "odds_below_minimum",
        "odds_above_maximum",
        "multiple_eligible_selections",
    }


def test_selection_assessment_is_immutable() -> None:
    assessment = SelectionAssessment(
        match_id="m01",
        selection_side="none",
        selection_player=None,
        eligible=False,
    )

    with pytest.raises(ValidationError):
        assessment.eligible = True  # type: ignore[misc]
