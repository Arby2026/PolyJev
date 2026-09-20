from decimal import Decimal as D

from chainlink_live import (
    ChainlinkLiveState,
    ChainlinkUpdate,
    choose_opening_reference,
    parse_rtds_event,
)


def raw(symbol="btc/usd", timestamp=1_700_000_000_000, value="65000.125"):
    return {
        "topic": "crypto_prices_chainlink",
        "type": "update",
        "payload": {"symbol": symbol, "timestamp": timestamp, "value": value},
    }


def twap(symbol="btc/usd", window=60, full="65000125000000000000000"):
    return {
        "topic": "crypto_prices_twap_sixty",
        "type": "update",
        "payload": {
            "symbol": symbol,
            "timestamp": 1_700_000_001_000,
            "value": "65000.1",
            "full_accuracy_value": full,
            "window_s": window,
        },
    }


def test_raw_chainlink_event_parsed():
    event = parse_rtds_event(raw(), "btc/usd")
    assert event is not None
    assert event.kind == "raw"
    assert event.value == D("65000.125")


def test_twap60_and_full_accuracy_parsed_exactly():
    event = parse_rtds_event(twap(full="12345678901234567890123"), "btc/usd")
    assert event is not None
    assert event.kind == "twap60"
    assert event.value == D("12345.678901234567890123")


def test_wrong_symbol_ignored():
    assert parse_rtds_event(raw(symbol="eth/usd"), "btc/usd") is None


def test_wrong_twap_window_ignored():
    assert parse_rtds_event(twap(window=30), "btc/usd") is None


def test_opening_reference_chooses_nearest_timestamp():
    updates = [
        ChainlinkUpdate("raw", "btc/usd", 995_000, D("99")),
        ChainlinkUpdate("raw", "btc/usd", 1_001_000, D("101")),
        ChainlinkUpdate("raw", "btc/usd", 1_004_000, D("104")),
    ]
    opening = choose_opening_reference(updates, 1_000_000)
    assert opening is not None
    assert opening.price == D("101")
    assert opening.offset_ms == 1000
    assert opening.quality == "good"


def test_poor_opening_reference_flagged():
    opening = choose_opening_reference(
        [ChainlinkUpdate("raw", "btc/usd", 1_006_000, D("101"))], 1_000_000
    )
    assert opening is not None
    assert opening.quality == "poor"


def test_chainlink_state_accepts_events_after_budget_independently():
    state = ChainlinkLiveState("btc/usd")
    assert state.apply(raw()) is True
    assert state.raw_events == 1
