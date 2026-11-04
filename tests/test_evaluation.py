from __future__ import annotations

import pytest

from icd_coder.catalogue import Catalogue
from icd_coder.evaluation import Example, evaluate, official_points


def example(note_id: str, gold: set[str], slice_name: str = "A") -> Example:
    return Example(note_id, "", frozenset(gold), slice_name, None)


def test_metrics_follow_the_scorer_rules(catalogue: Catalogue) -> None:
    examples = [
        example("hit", {"1F40.Z"}),
        example("second", {"5A11"}),
        example("invalid", {"1D80.0"}),
        example("missing", {"1D80.0"}),
    ]
    predictions = {
        "hit": ["1F40.Z", "1F40.0"],
        "second": ["1F40.Z", "5A11&XA1"],
        "invalid": ["NOT.A.CODE", "1D80.0", "1D80.1", "5A11"],
    }
    metrics = evaluate(examples, predictions, catalogue)["A"]
    assert metrics.top1 == pytest.approx(1 / 4)
    assert metrics.top3 == pytest.approx(3 / 4)
    assert metrics.mrr == pytest.approx((1 + 1 / 2 + 1 / 2) / 4)
    assert metrics.missing_rate == pytest.approx(1 / 4)
    assert metrics.invalid_rate == pytest.approx(1 / 12)


def test_family_metric_counts_wrong_specificity_as_near_miss(catalogue: Catalogue) -> None:
    metrics = evaluate([example("n", {"CA23.30"})], {"n": ["CA23.32"]}, catalogue)["A"]
    assert metrics.top1 == 0
    assert metrics.family_top1 == 1


def test_points_use_the_slice_weights(catalogue: Catalogue) -> None:
    examples = [example("a", {"5A11"}), example("b", {"5A11"}, "B")]
    metrics = evaluate(examples, {"a": ["5A11"], "b": ["1F40.Z", "5A11"]}, catalogue)
    earned, available = official_points(metrics)
    assert available == 100
    assert earned == pytest.approx(75 + 25 / 3)
