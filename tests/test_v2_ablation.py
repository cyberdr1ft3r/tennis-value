from __future__ import annotations

import pandas as pd
from tests.test_train_v2 import _features

from tennis_value.diagnose_v2 import ABLATION_VARIANTS, run_ablation_study
from tennis_value.train import MODEL_FEATURES as MODEL_V1_FEATURES
from tennis_value.train_v2 import (
    FORBIDDEN_FEATURES,
    MARKET_FEATURES,
    build_walk_forward_folds,
    prepare_model_v2_dataset,
)


def test_ablation_feature_allowlists_exclude_leakage_columns() -> None:
    for spec in ABLATION_VARIANTS.values():
        features = set(spec["features"])
        assert FORBIDDEN_FEATURES.isdisjoint(features)
        assert {"edge", "expected_value", "stake", "bankroll_after", "profit_loss"}.isdisjoint(
            features
        )
        if features:
            assert "market_logit_player_1" in features
            assert features.issubset(set(MARKET_FEATURES))
    assert "market_logit_player_1" not in MODEL_V1_FEATURES


def test_ablation_uses_strict_walk_forward_boundaries() -> None:
    prepared = prepare_model_v2_dataset(_features())
    folds = build_walk_forward_folds(prepared)

    for fold in folds:
        assert fold.train["match_date"].max() < fold.evaluation["match_date"].min()
        assert set(fold.train["match_id"]).isdisjoint(set(fold.evaluation["match_id"]))


def test_ablation_outputs_metrics_and_coefficients() -> None:
    metrics, coefficients = run_ablation_study(_features())

    assert set(metrics["variant"]) == set(ABLATION_VARIANTS)
    assert set(metrics["evaluation_year"]) == {2023, 2024, 2025}
    assert {"variant", "evaluation_year", "model_log_loss", "market_log_loss"}.issubset(
        metrics.columns
    )
    assert {"feature_name", "coefficient", "sign", "absolute_coefficient"}.issubset(
        coefficients.columns
    )
    assert "market_logit_player_1" in coefficients["feature_name"].tolist()


def test_ablation_input_dataframe_is_not_mutated() -> None:
    features = _features()
    original = features.copy(deep=True)

    run_ablation_study(features)

    pd.testing.assert_frame_equal(features, original)
