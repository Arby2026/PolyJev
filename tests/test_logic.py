import math
import statistics
from datetime import datetime, timezone

import pytest

from features import calculate_features, calculate_p_simple
from market_data import Candle
from polymarket import (
    best_prices,
    map_outcome_tokens,
    market_slug,
    parse_resolved_outcome,
    window_start_for_timestamp,
)
from run import checkpoint_due
from storage import save_snapshot, snapshot_exists


def test_timestamp_to_window_start():
    assert window_start_for_timestamp(1789910778) == 1789910100
    assert window_start_for_timestamp(1789910100) == 1789910100


def test_slug_generation():
    assert market_slug("BTC", 1789910100) == "btc-updown-15m-1789910100"
    assert market_slug("ETH", 1789910100) == "eth-updown-15m-1789910100"


def test_outcome_token_mapping_from_json_strings():
    assert map_outcome_tokens('["Down", "Up"]', '["down-id", "up-id"]') == {
        "Up": "up-id",
        "Down": "down-id",
    }
    with pytest.raises(RuntimeError, match="Up/Down"):
        map_outcome_tokens('["Yes", "No"]', '["1", "2"]')


def test_best_bid_and_ask_do_not_depend_on_array_order():
    book = {
        "bids": [{"price": "0.45"}, {"price": "0.51"}, {"price": "0.48"}],
        "asks": [{"price": "0.58"}, {"price": "0.53"}, {"price": "0.55"}],
    }
    assert best_prices(book) == (0.51, 0.53)


def test_feature_calculation_on_synthetic_candles():
    candles = [
        Candle(
            open_time_ms=i * 60_000,
            close_time_ms=(i + 1) * 60_000 - 1,
            open=100.0 + i,
            close=101.0 + i,
        )
        for i in range(21)
    ]
    result = calculate_features(candles, 122.0, 900, 1800, 1260.0)
    assert result["proxy_start_price"] == 115.0
    assert result["distance_from_start_bps"] == pytest.approx((122 / 115 - 1) * 10_000)
    assert result["return_1m"] == pytest.approx(122 / 120 - 1)
    assert result["return_5m"] == pytest.approx(122 / 116 - 1)
    expected_returns = [math.log(b / a) for a, b in zip(range(116, 121), range(117, 122))]
    assert result["realized_vol_5m"] == pytest.approx(statistics.pstdev(expected_returns))
    assert result["time_remaining_sec"] == 540.0


def test_checkpoint_trigger_and_missed_logic():
    assert checkpoint_due(300, 300)
    assert checkpoint_due(285, 300)
    assert not checkpoint_due(301, 300)
    assert not checkpoint_due(284.9, 300)


def test_p_simple_direction_and_zero_distance():
    assert calculate_p_simple(100, 101, 0.01, 300) > 0.5
    assert calculate_p_simple(100, 99, 0.01, 300) < 0.5
    assert calculate_p_simple(100, 100, 0.01, 300) == pytest.approx(0.5)
    assert calculate_p_simple(100, 101, 0.0, 300) == 1.0
    assert calculate_p_simple(100, 99, 0.0, 300) == 0.0
    assert calculate_p_simple(100, 100, 0.0, 300) == 0.5
    assert calculate_p_simple(100, 101, 0.01, 0) is None


@pytest.mark.parametrize(
    ("outcomes", "prices", "expected"),
    [
        ('["Up", "Down"]', '["1", "0"]', "UP"),
        ('["Up", "Down"]', '["0", "1"]', "DOWN"),
        ('["Up", "Down"]', '["0.55", "0.45"]', None),
    ],
)
def test_resolved_outcome_parsing(outcomes, prices, expected):
    assert parse_resolved_outcome(outcomes, prices) == expected


def test_duplicate_detection(tmp_path):
    database = str(tmp_path / "test.duckdb")
    snapshot = {
        "snapshot_id": "00000000-0000-0000-0000-000000000001",
        "observed_at": datetime.now(timezone.utc),
        "asset": "BTC",
        "market_slug": "btc-updown-15m-1789910100",
        "market_id": "market",
        "condition_id": "condition",
        "window_start": 1789910100,
        "window_end": 1789911000,
        "rules": "Chainlink TWAP rules",
        "resolution_source": "Chainlink BTC/USD TWAP",
        "up_token_id": "up",
        "down_token_id": "down",
        "up_best_bid": 0.49,
        "up_best_ask": 0.51,
        "down_best_bid": 0.49,
        "down_best_ask": 0.51,
        "up_last_trade_price": 0.5,
        "down_last_trade_price": 0.5,
        "p_market": 0.5,
        "proxy_source": "Binance Spot REST (predictive proxy only)",
        "proxy_start_price": 100.0,
        "proxy_current_price": 101.0,
        "distance_from_start_bps": 100.0,
        "return_1m": 0.001,
        "return_5m": 0.002,
        "realized_vol_5m": 0.003,
        "realized_vol_15m": 0.004,
        "time_remaining_sec": 300.0,
        "raw_market_json": "{}",
        "checkpoint": "T-5",
        "p_simple": 0.75,
        "outcome": None,
        "resolved_at": None,
    }
    assert not snapshot_exists(database, snapshot["market_slug"], "T-5")
    save_snapshot(database, snapshot)
    assert snapshot_exists(database, snapshot["market_slug"], "T-5")
