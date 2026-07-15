from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from tennis_value.config import DateSplitConfig, TrainingConfig
from tennis_value.train import (
    BINARY_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    build_model_pipeline,
    dataset_sha256,
    split_feature_dataset,
    train_probability_model,
    write_training_outputs,
)


def _row(
    match_id: str,
    match_date: str,
    target: bool,
    *,
    is_retirement: bool = False,
    value: float = 0.0,
    log_rank_diff: float | None = 0.0,
) -> dict[str, object]:
    return {
        "match_id": match_id,
        "match_date": pd.Timestamp(match_date),
        "surface": "Hard",
        "player_1": f"Player {match_id} A",
        "player_2": f"Player {match_id} B",
        "player_1_won": target,
        "player_1_odds": 1.8,
        "player_2_odds": 2.0,
        "is_retirement": is_retirement,
        "overall_elo_diff": value,
        "surface_elo_diff": value / 2,
        "log_rank_diff": log_rank_diff,
        "recent_10_win_rate_diff": value / 100,
        "surface_recent_10_win_rate_diff": value / 100,
        "days_since_last_match_diff": None if log_rank_diff is None else value,
        "matches_last_14d_diff": int(value) % 3,
        "history_count_min": int(abs(value)) % 5,
        "best_of_5": 0,
        "surface_clay": 0,
        "surface_grass": 0,
    }


def _training_frame() -> pd.DataFrame:
    rows = [
        _row("tr01", "2020-01-01", True, value=1.0, log_rank_diff=1.0),
        _row("tr02", "2021-06-01", False, value=2.0, log_rank_diff=2.0),
        _row("tr03", "2022-06-01", True, value=3.0, log_rank_diff=None),
        _row("tr04", "2023-12-31", False, value=4.0, log_rank_diff=4.0),
        _row("va01", "2024-01-01", True, value=10000.0, log_rank_diff=10000.0),
        _row("va02", "2024-12-31", False, value=11000.0, log_rank_diff=None),
        _row("te01", "2025-01-01", True, value=-10000.0, log_rank_diff=-10000.0),
        _row("te02", "2025-12-31", False, value=-11000.0, log_rank_diff=None),
        _row("out01", "2019-12-31", True, value=99.0),
        _row("ret01", "2021-01-01", True, is_retirement=True, value=99.0),
    ]
    return pd.DataFrame(rows)


def test_chronological_split_boundaries_and_no_overlap() -> None:
    partitions = split_feature_dataset(_training_frame(), DateSplitConfig())

    assert partitions.train["match_id"].tolist() == ["tr01", "tr02", "tr03", "tr04"]
    assert partitions.validation["match_id"].tolist() == ["va01", "va02"]
    assert partitions.test["match_id"].tolist() == ["te01", "te02"]
    assert set(partitions.train["match_id"]).isdisjoint(partitions.validation["match_id"])
    assert set(partitions.train["match_id"]).isdisjoint(partitions.test["match_id"])
    assert set(partitions.validation["match_id"]).isdisjoint(partitions.test["match_id"])


def test_input_order_does_not_affect_partitions_or_predictions() -> None:
    frame = _training_frame()
    shuffled = frame.sample(frac=1, random_state=7).reset_index(drop=True)

    first = train_probability_model(frame)
    second = train_probability_model(shuffled)

    pd.testing.assert_frame_equal(first.predictions, second.predictions)


def test_model_feature_allowlist_excludes_leakage_columns() -> None:
    assert "player_1_odds" not in MODEL_FEATURES
    assert "player_2_odds" not in MODEL_FEATURES
    assert "player_1_won" not in MODEL_FEATURES
    assert "match_id" not in MODEL_FEATURES
    assert "match_date" not in MODEL_FEATURES
    assert "player_1" not in MODEL_FEATURES
    assert "player_2" not in MODEL_FEATURES

    with pytest.raises(ValueError, match="MODEL_FEATURES"):
        build_model_pipeline([*MODEL_FEATURES, "player_1_odds"])


def test_missing_required_feature_columns_fail_helpfully() -> None:
    with pytest.raises(ValueError, match="missing required training input columns"):
        train_probability_model(_training_frame().drop(columns=["overall_elo_diff"]))


def test_preprocessing_fits_only_on_training_rows() -> None:
    result = train_probability_model(_training_frame())
    preprocessor = result.model.named_steps["preprocessor"]
    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    imputer = numeric_pipeline.named_steps["imputer"]
    scaler = numeric_pipeline.named_steps["scaler"]

    log_rank_index = NUMERIC_FEATURES.index("log_rank_diff")
    assert imputer.statistics_[log_rank_index] == pytest.approx(2.0)
    assert max(abs(value) for value in imputer.statistics_) < 100
    assert scaler.mean_[log_rank_index] == pytest.approx(2.25)
    assert max(abs(value) for value in scaler.mean_[: len(NUMERIC_FEATURES)]) < 100
    assert imputer.indicator_.features_.tolist()


def test_prediction_handles_missing_values_in_validation_and_test() -> None:
    result = train_probability_model(_training_frame())

    assert set(result.predictions["partition"]) == {"train", "validation", "test"}
    assert result.predictions["predicted_player_1_probability"].between(0, 1).all()
    assert result.predictions["predicted_player_2_probability"].between(0, 1).all()
    assert (
        result.predictions["predicted_player_1_probability"]
        + result.predictions["predicted_player_2_probability"]
    ).tolist() == pytest.approx([1.0] * len(result.predictions))


def test_single_class_training_data_raises() -> None:
    frame = _training_frame()
    frame.loc[frame["match_id"].str.startswith("tr"), "player_1_won"] = True

    with pytest.raises(ValueError, match="both target classes"):
        train_probability_model(frame)


def test_retirements_are_excluded_from_fit_and_summary() -> None:
    result = train_probability_model(_training_frame())

    assert "ret01" not in set(result.partitions.train["match_id"])
    assert "ret01" not in set(result.predictions["match_id"])
    assert result.summary.excluded_retirements == 1


def test_fixed_c_is_recorded() -> None:
    result = train_probability_model(_training_frame(), TrainingConfig(classifier_c=0.5))

    classifier = result.model.named_steps["classifier"]
    assert classifier.C == 0.5
    assert result.metadata.selected_c == 0.5
    assert result.metadata.selection_metric == "fixed_C_no_validation_selection"


def test_artifacts_are_written_and_loaded_model_predicts_the_same() -> None:
    artifact_dir = Path(".tmp-task7-artifacts")
    artifact_dir.mkdir(exist_ok=True)
    input_path = artifact_dir / "features.parquet"
    model_path = artifact_dir / "model.joblib"
    metadata_path = artifact_dir / "metadata.json"
    predictions_path = artifact_dir / "predictions.parquet"
    summary_path = artifact_dir / "summary.json"
    frame = _training_frame()
    frame.to_parquet(input_path, index=False)

    result = train_probability_model(
        frame,
        input_dataset_path=input_path,
        model_output_path=model_path,
        prediction_output_path=predictions_path,
    )
    write_training_outputs(result, model_path, metadata_path, predictions_path, summary_path)

    loaded = joblib.load(model_path)
    loaded_probabilities = loaded.predict_proba(result.partitions.test[MODEL_FEATURES])[:, 1]
    original_test = result.predictions[result.predictions["partition"] == "test"]
    assert loaded_probabilities.tolist() == pytest.approx(
        original_test["predicted_player_1_probability"].tolist()
    )
    json.loads(metadata_path.read_text(encoding="utf-8"))
    json.loads(summary_path.read_text(encoding="utf-8"))
    assert dataset_sha256(input_path) == result.metadata.input_dataset_sha256
    assert pd.read_parquet(predictions_path).columns.tolist() == result.predictions.columns.tolist()


def test_binary_features_are_validated() -> None:
    frame = _training_frame()
    frame.loc[0, BINARY_FEATURES[0]] = 2

    with pytest.raises(ValueError, match="binary"):
        train_probability_model(frame)
