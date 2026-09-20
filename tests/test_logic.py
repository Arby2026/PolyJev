import math
import statistics

import pytest

from features import calculate_features
from market_data import Candle
from polymarket import best_prices, map_outcome_tokens, market_slug, window_start_for_timestamp


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

