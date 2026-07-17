from __future__ import annotations

from tests.test_train_v2 import _features

from tennis_value.train_v2 import build_walk_forward_folds, prepare_model_v2_dataset


def test_walk_forward_boundaries_are_strict() -> None:
    prepared = prepare_model_v2_dataset(_features())
    folds = build_walk_forward_folds(prepared)

    expected = [
        (2020, 2022, 2023),
        (2020, 2023, 2024),
        (2020, 2024, 2025),
    ]
    for fold, (start, train_end, evaluation_year) in zip(folds, expected, strict=True):
        train = fold.train
        evaluation = fold.evaluation
        assert train["year"].min() == start
        assert train["year"].max() == train_end
        assert evaluation["year"].unique().tolist() == [evaluation_year]
        assert train["match_date"].max() < evaluation["match_date"].min()


def test_no_row_appears_in_its_own_training_period() -> None:
    prepared = prepare_model_v2_dataset(_features())
    folds = build_walk_forward_folds(prepared)

    for fold in folds:
        train = fold.train
        evaluation = fold.evaluation
        assert set(train["match_id"]).isdisjoint(set(evaluation["match_id"]))
