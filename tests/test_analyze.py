import math

import pytest

from analyze import (
    AnalysisError,
    binary_log_loss,
    brier_score,
    disagreement_metrics,
    ensure_no_duplicate_complete_rows,
    outcome_to_binary,
)


def test_brier_formula():
    assert brier_score([0.8, 0.3], [1, 0]) == pytest.approx(0.065)


def test_log_loss_formula():
    expected = -(math.log(0.8) + math.log(0.75)) / 2
    assert binary_log_loss([0.8, 0.25], [1, 0]) == pytest.approx(expected)


def test_disagreement_subset_and_delta_brier():
    result = disagreement_metrics(
        market_probabilities=[0.60, 0.40, 0.52],
        jev_probabilities=[0.70, 0.35, 0.55],
        outcomes=[1, 0, 1],
        threshold=0.05,
    )
    assert result["n"] == 2
    assert result["mean_absolute_disagreement"] == pytest.approx(0.075)
    assert result["market_brier"] == pytest.approx(0.16)
    assert result["jev_brier"] == pytest.approx(0.10625)
    assert result["delta_brier"] == pytest.approx(0.05375)


def test_duplicate_detection():
    row = {
        "asset": "BTC",
        "market_slug": "btc-updown-15m-test",
        "checkpoint": "T-5",
    }
    with pytest.raises(AnalysisError, match="duplicate complete observations"):
        ensure_no_duplicate_complete_rows([row, row.copy()])


@pytest.mark.parametrize(("outcome", "expected"), [("UP", 1), ("DOWN", 0)])
def test_outcome_to_binary(outcome, expected):
    assert outcome_to_binary(outcome) == expected
