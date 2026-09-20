import time
from decimal import Decimal as D

from chainlink_live import ChainlinkLiveState
from forecast_experiment import (
    CostGuard,
    PendingLabel,
    build_compact_state,
    build_questions,
    evaluate_label,
    hold_to_resolution_pnl,
    hypothetical_entries,
    terminal_break_even,
    terminal_edges,
)
from live_market import LiveMarketState
from live_trader import MarketInfo
from paper_trader import FeeSchedule, Level, quote_buy


ENABLED = FeeSchedule(True, D("0.07"), D("1"), True, D("0.2"))


def ready_clob(size="100", up_ask="0.50", down_ask="0.50"):
    state = LiveMarketState("up", "down")
    for token, ask in (("up", up_ask), ("down", down_ask)):
        state.apply_event(
            {
                "event_type": "book",
                "asset_id": token,
                "bids": [{"price": "0.49", "size": size}],
                "asks": [{"price": ask, "size": size}],
            }
        )
    return state


def chainlink_state():
    state = ChainlinkLiveState("btc/usd")
    state.apply(
        {
            "topic": "crypto_prices_chainlink",
            "type": "update",
            "payload": {"symbol": "btc/usd", "timestamp": 1_000_000, "value": "65000"},
        }
    )
    state.apply(
        {
            "topic": "crypto_prices_twap_sixty",
            "type": "update",
            "payload": {
                "symbol": "btc/usd",
                "timestamp": 1_001_000,
                "value": "65001",
                "full_accuracy_value": "65001000000000000000000",
                "window_s": 60,
            },
        }
    )
    state.capture_opening(1_000_000)
    return state


def market():
    return MarketInfo(
        asset="BTC",
        slug="btc-updown-15m-0",
        window_start=0,
        window_end=900,
        rules="UP iff Chainlink TWAP for this contract range >= start price; else DOWN",
        resolution_source="Chainlink",
        up_token="up",
        down_token="down",
        minimum_tick_size=D("0.01"),
        minimum_order_size=D("5"),
        fee_schedule=ENABLED,
        raw_market={},
    )


def test_compact_state_has_chainlink_clob_economics_and_no_position_history():
    clob = ready_clob()
    compact = build_compact_state(
        market(), chainlink_state(), clob, hypothetical_entries(clob, ENABLED), now=100
    )
    assert {"chainlink", "up", "down", "economics"} <= set(compact)
    serialized = str(compact).lower()
    assert "position" not in serialized
    assert "previous" not in serialized
    assert "answer" not in serialized


def test_questions_omit_horizon_crossing_market_end_but_keep_terminal():
    entries = hypothetical_entries(ready_clob(), ENABLED)
    questions = build_questions(31, entries)
    assert "resolve_up" in questions
    assert "up_profit_15s" in questions
    assert "up_profit_30s" not in questions
    assert "up_profit_60s" not in questions


def test_unexecutable_entry_omits_side_questions():
    entries = hypothetical_entries(ready_clob(size="1"), ENABLED)
    questions = build_questions(500, entries)
    assert list(questions) == ["resolve_up"]


def pending(entry, horizon=15):
    now = time.monotonic()
    return PendingLabel("id", "UP", horizon, 0.7, 100.0, 115.0, now, entry)


def test_profitable_label_after_fees_uses_future_bids():
    entry = quote_buy([Level(D("0.40"), D("100"))], D("10"), ENABLED)
    result = evaluate_label(
        pending(entry), [Level(D("0.60"), D("100"))], ENABLED,
        115.0, time.monotonic(), time.monotonic(), 900,
    )
    assert result.status == "valid"
    assert result.profitable is True
    assert result.exit_quote.fills[0].price == D("0.60")
    assert entry.fills[0].price == D("0.40")


def test_negative_label_after_fees():
    entry = quote_buy([Level(D("0.50"), D("100"))], D("10"), ENABLED)
    result = evaluate_label(
        pending(entry), [Level(D("0.50"), D("100"))], ENABLED,
        115.0, time.monotonic(), time.monotonic(), 900,
    )
    assert result.status == "valid"
    assert result.profitable is False
    assert result.actual_net_pnl < 0


def test_unexecutable_exit_is_not_false():
    entry = quote_buy([Level(D("0.50"), D("100"))], D("10"), ENABLED)
    result = evaluate_label(
        pending(entry), [Level(D("0.60"), D("1"))], ENABLED,
        115.0, time.monotonic(), time.monotonic(), 900,
    )
    assert result.status == "unexecutable"
    assert result.profitable is None


def test_stale_future_book():
    entry = quote_buy([Level(D("0.50"), D("100"))], D("10"), ENABLED)
    now = time.monotonic()
    result = evaluate_label(
        pending(entry), [Level(D("0.60"), D("100"))], ENABLED,
        115.0, now, now - 3, 900,
    )
    assert result.status == "stale"
    assert result.profitable is None


def test_terminal_break_even_and_edges():
    entries = hypothetical_entries(ready_clob(up_ask="0.25", down_ask="0.75"), ENABLED)
    up_break_even = terminal_break_even(entries["UP"])
    assert up_break_even == entries["UP"].net_cash / entries["UP"].shares
    edges = terminal_edges(0.70, entries)
    assert edges["model_terminal_edge_up"] == D("0.70") - up_break_even
    assert edges["model_terminal_edge_down"] == D("0.30") - terminal_break_even(entries["DOWN"])


def test_winning_and_losing_hold_to_resolution_pnl():
    entry = quote_buy([Level(D("0.25"), D("100"))], D("10"), ENABLED)
    assert hold_to_resolution_pnl(entry, "DOWN", "DOWN") == entry.shares - entry.net_cash
    assert hold_to_resolution_pnl(entry, "DOWN", "UP") == -entry.net_cash


def test_token_cost_accumulator_and_budget_guard():
    guard = CostGuard(D("0.000042"))
    cost = guard.record(1000)
    assert cost == D("0.000042")
    assert guard.estimated_cost == cost
    assert guard.can_request is False
    assert guard.metrics()["input_tokens_total"] == 1000
