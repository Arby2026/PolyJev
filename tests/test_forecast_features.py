import math
from decimal import Decimal as D

import pytest

from chainlink_live import ChainlinkUpdate
from forecast_features import RawVolatility, TradeFlow, book_features, detect_wall, gate_reason, trades_subscription
from forecast_report import pearson
from live_market import LiveMarketState
from paper_trader import Level


def trade(timestamp=100, token="up", price=0.5, size=90, **extra):
    return {"topic": "activity", "type": "trades", "payload": {
        "slug": "market", "asset": token, "side": "BUY", "timestamp": timestamp,
        "price": price, "size": size, **extra,
    }}


def test_flow_uses_dollar_notional_and_filters_market_side_and_duplicate():
    flow = TradeFlow("market", "up", "down")
    flow.connected_since = 30
    assert flow.apply(trade(transactionHash="tx", proxyWallet="a"), 100)
    assert not flow.apply(trade(transactionHash="tx", proxyWallet="a"), 100)
    assert flow.apply(trade(token="down", price=0.4, size=50), 100)
    for event in (trade(slug="another"), trade(token="unknown"), trade(side="SELL"), trade(size=-1), trade(price=float("nan")), trade(timestamp=101)):
        assert not flow.apply(event, 100)
    result = flow.snapshot(100)
    assert result["up_count"] == result["down_count"] == 1
    assert result["net"] == 0
    assert result["avg_size"] == 32.5
    assert result["imbalance_usd"] == 25
    assert result["status"] == "ready"
    assert trades_subscription()["subscriptions"][0] == {
        "topic": "activity", "type": "trades",
    }


def test_flow_expires_out_of_order_ticks_even_without_new_trades():
    flow = TradeFlow("market", "up", "down")
    assert flow.apply(trade(timestamp=100), 100)
    assert flow.apply(trade(timestamp=50, token="down"), 100)
    assert flow.snapshot(111)["down_count"] == 0
    assert flow.snapshot(161)["up_count"] == 0
    assert flow.snapshot(161)["status"] == "unavailable"
    flow.connected_since = 160
    assert flow.snapshot(161)["status"] == "warming_up"


def test_flow_accepts_epoch_milliseconds_and_preserves_distinct_fills_in_transaction():
    flow = TradeFlow("market", "up", "down")
    now = 1_700_000_000
    assert flow.apply(trade(timestamp=now * 1000, transactionHash="tx", size=1), now)
    assert flow.apply(trade(timestamp=now * 1000, transactionHash="tx", size=2), now)
    assert flow.snapshot(now)["up_count"] == 2


@pytest.mark.parametrize("mid,expected", [(0.099, "mid_below_0.10"), (0.1, None), (0.5, None), (0.9, None), (0.901, "mid_above_0.90")])
def test_gate_strict_boundaries(mid, expected):
    assert gate_reason(mid) == expected


def test_book_units_depth_and_walls_for_both_outcomes():
    clob = LiveMarketState("up", "down")
    for token, multiplier in (("up", 4), ("down", 1)):
        clob.apply_event({"event_type": "book", "asset_id": token,
            "bids": [{"price": "0.50", "size": str(100 * multiplier)}],
            "asks": [{"price": "0.51", "size": str(200 * multiplier)}],
        })
    values = book_features(clob)
    assert values["book"]["up"]["bid_usd"] == 200
    assert values["book"]["up"]["ask_usd"] == 408
    assert values["spread_bps"] == pytest.approx(0.01 / 0.505 * 10000)
    assert values["depth_ratio"] == 4
    assert values["wall_text"] == "none detected"
    for outcome in ("UP", "DOWN"):
        for side in ("ASK", "BID"):
            levels = [Level(D("0.5"), D(10400))] + [Level(D("0.6"), D(1000))] * 5
            assert detect_wall(levels, D(9000), outcome, side) == f"{outcome} {side} wall 0.5 $5200 (10.4x)"


def test_wall_requires_both_strict_thresholds_and_five_comparison_levels():
    others = [Level(D("0.6"), D(1000))] * 5
    assert detect_wall([Level(D("0.5"), D(3000))] + others, D(6000), "UP", "ASK") is None
    assert detect_wall([Level(D("0.9"), D(3000))] + others, D(6000), "UP", "ASK") is None
    assert detect_wall([Level(D("0.5"), D(4000))] + others, D(8000), "UP", "ASK") is None
    assert detect_wall([Level(D("0.5"), D(4000))] + others[:4], D(5000), "UP", "ASK") is None


def raw(timestamp, value):
    return ChainlinkUpdate("raw", "btc/usd", int(timestamp * 1000), D(str(value)))


def test_volatility_log_returns_percent_reversals_and_expiry():
    vol = RawVolatility()
    for i, value in enumerate((100, 101, 100, 101)):
        vol.add(raw(100 + i, value), 103)
    snap = vol.snapshot(103)
    assert snap["vol_60s"] == pytest.approx(math.sqrt(8 / 9) * math.log(1.01) * 100)
    assert snap["vol_direction_changes"] == 2
    assert snap["vol_regime"] == "choppy"
    vol.add(raw(102, 100), 103)  # Replayed/out-of-order sample.
    assert vol.snapshot(103) == snap
    assert vol.snapshot(164)["vol_60s"] is None


def test_volatility_retains_more_than_100_ticks_and_ignores_twap():
    vol = RawVolatility()
    for i in range(200):
        vol.add(raw(100 + i / 10, 100 + i / 100), 120)
    vol.add(ChainlinkUpdate("twap60", "btc/usd", 120000, D(1000)), 120)
    assert vol.snapshot(120)["vol_samples"] == 200
    assert vol.snapshot(120)["vol_regime"] == "trend"


def test_pearson_handles_constants_missing_nonfinite_and_negative_dependence():
    assert pearson([(0.1, 1), (0.2, 2), (0.3, 3)])[0] == pytest.approx(1)
    assert pearson([(0.1, 3), (0.2, 2), (0.3, 1)])[0] == pytest.approx(-1)
    assert pearson([(0.1, 1)] * 3) == (None, 3)
    assert pearson([(0.1, None), (0.2, float("nan")), (0.3, 3)]) == (None, 1)
