import json
import time
from decimal import Decimal as D

import pytest

from chainlink_live import ChainlinkLiveState
from forecast_experiment import (
    CostGuard,
    PendingLabel,
    build_compact_state,
    build_questions,
    capture_realtime_state,
    classify_winner_loser_edges,
    create_pending_labels,
    detect_transitions,
    disagreement_side_pnl,
    evaluate_label,
    hold_to_resolution_pnl,
    hypothetical_entries,
    normalized_market_probabilities,
    probability_sides,
    select_best_edge,
    signed_repricing,
    terminal_break_even,
    terminal_edges,
    winner_relative_metrics,
)
from forecast_report import generate_report, load_records
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
    assert {"chainlink", "up", "down", "terminal_economics"} <= set(compact)
    serialized = str(compact).lower()
    assert "position" not in serialized
    assert "previous" not in serialized
    assert "answer" not in serialized


def test_exactly_one_jev_question_resolve_up():
    entries = hypothetical_entries(ready_clob(), ENABLED)
    questions = build_questions(31, entries)
    assert questions == {
        "resolve_up": "Will this contract resolve UP under the stated Chainlink rule?"
    }


def test_unexecutable_entry_omits_side_questions():
    entries = hypothetical_entries(ready_clob(size="1"), ENABLED)
    questions = build_questions(500, entries)
    assert list(questions) == ["resolve_up"]


def test_p_down_is_complement_of_p_up():
    values = probability_sides(0.73, D("0.55"))
    assert values["p_jev_down"] == D("0.27")
    assert values["probability_delta_down"] == -values["probability_delta_up"]


def test_normalized_market_probability_and_sum():
    result = normalized_market_probabilities(D("0.54"), D("0.56"), D("0.43"), D("0.45"))
    assert result["up_mid"] == D("0.55")
    assert result["down_mid"] == D("0.44")
    assert result["p_market_up"] + result["p_market_down"] == D("1")


def test_normalized_market_probability_is_symmetric_when_sides_swapped():
    original = normalized_market_probabilities(D("0.54"), D("0.56"), D("0.43"), D("0.45"))
    swapped = normalized_market_probabilities(D("0.43"), D("0.45"), D("0.54"), D("0.56"))
    assert original["p_market_up"] == swapped["p_market_down"]
    assert original["p_market_down"] == swapped["p_market_up"]


def test_response_state_and_entry_use_latest_response_asks():
    chain = chainlink_state()
    input_clob = ready_clob(up_ask="0.50")
    response_clob = ready_clob(up_ask="0.80")
    input_entries = hypothetical_entries(input_clob, ENABLED)
    response_entries = hypothetical_entries(response_clob, ENABLED)
    input_state = capture_realtime_state(market(), chain, input_clob, input_entries, 100)
    response_state = capture_realtime_state(market(), chain, response_clob, response_entries, 102)
    assert response_state["timestamp"] > input_state["timestamp"]
    assert input_state["up"]["ask"] == "0.50"
    assert response_state["up"]["ask"] == "0.80"
    assert response_entries["UP"].fills[0].price == D("0.80")


def test_future_horizons_begin_at_response_timestamp():
    entries = hypothetical_entries(ready_clob(), ENABLED)
    pending_labels = create_pending_labels("id", 200.0, 300.0, 0.5, 0.6, entries)
    assert [item.target_epoch for item in pending_labels] == [215.0, 230.0, 260.0, 320.0]
    assert [item.due_monotonic for item in pending_labels] == [315.0, 330.0, 360.0, 420.0]


def test_best_edge_side_up_down_and_none():
    assert select_best_edge({"model_terminal_edge_up": D("0.1"), "model_terminal_edge_down": D("-0.2")})["best_edge_side"] == "UP"
    assert select_best_edge({"model_terminal_edge_up": D("-0.1"), "model_terminal_edge_down": D("0.2")})["best_edge_side"] == "DOWN"
    assert select_best_edge({"model_terminal_edge_up": D("-0.1"), "model_terminal_edge_down": D("-0.2")})["best_edge_side"] == "NONE"


def test_winner_relative_metrics_symmetric_for_up_and_down():
    up = {"p_jev_up": 0.7, "response_market_probabilities": {"p_market_up": "0.6"}}
    down = {"p_jev_up": 0.3, "response_market_probabilities": {"p_market_up": "0.4"}}
    up_metrics = winner_relative_metrics(up, "UP")
    down_metrics = winner_relative_metrics(down, "DOWN")
    assert up_metrics.keys() == down_metrics.keys()
    for key in up_metrics:
        assert up_metrics[key] == pytest.approx(down_metrics[key])


def test_signed_repricing_is_symmetric_for_up_and_down_disagreement():
    assert signed_repricing(0.7, 0.5, 0.6) == pytest.approx(0.1)
    assert signed_repricing(0.3, 0.5, 0.4) == pytest.approx(0.1)


def test_disagreement_side_pnl_selects_correct_side():
    assert disagreement_side_pnl(0.7, 0.5, D("1"), D("-1")) == D("1")
    assert disagreement_side_pnl(0.3, 0.5, D("1"), D("-1")) == D("-1")


def test_winner_loser_edge_classification():
    assert classify_winner_loser_edges("UP", D("0.1"), D("-0.1")) == "WINNER_EDGE_ONLY"
    assert classify_winner_loser_edges("DOWN", D("0.1"), D("-0.1")) == "LOSER_EDGE_ONLY"
    assert classify_winner_loser_edges("DOWN", D("0.1"), D("0.2")) == "BOTH"
    assert classify_winner_loser_edges("UP", D("-0.1"), D("-0.2")) == "NEITHER"


def test_all_sources_detect_both_crossover_directions():
    values = [(1.0, "UP", 800), (2.0, "DOWN", 700), (3.0, "UP", 600)]
    for source in ("RAW", "TWAP", "JEV", "MARKET"):
        transitions = detect_transitions(values, source)
        assert [(item["from_side"], item["to_side"]) for item in transitions] == [
            ("UP", "DOWN"),
            ("DOWN", "UP"),
        ]


def test_legacy_jsonl_still_loads_and_reports(tmp_path):
    path = tmp_path / "legacy.jsonl"
    records = [
        {"record_type": "session_start", "asset": "BTC", "window_start": 0},
        {"record_type": "market_resolved", "winner": "UP"},
    ]
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    assert len(load_records(path)) == 2
    report = generate_report(path, resolve_missing=False)
    assert "actionable/latency-adjusted metrics = unavailable" in report


def test_v2_side_neutral_report_renders(tmp_path):
    path = tmp_path / "v2.jsonl"
    forecast = {
        "record_type": "forecast",
        "experiment_version": 2,
        "decision_id": "d1",
        "response_timestamp": 100.0,
        "time_remaining": 700,
        "p_jev_up": 0.7,
        "response_market_probabilities": {"p_market_up": "0.6"},
        "jev_side": "UP",
        "market_side": "UP",
        "raw_side": "DOWN",
        "twap_side": "UP",
        "best_edge_side": "UP",
        "actionable_edge_up": "0.1",
        "actionable_edge_down": "-0.2",
        "terminal_break_even_up": "0.6",
        "terminal_break_even_down": "0.4",
        "response_state": {
            "up": {"bid": "0.58", "ask": "0.60"},
            "down": {"bid": "0.40", "ask": "0.42"},
            "chainlink": {"raw_from_open_bps": 1.0, "twap_from_open_bps": 0.5},
        },
        "response_entries": {
            "UP": {"shares": "16.66", "net_cash": "10.2"},
            "DOWN": {"shares": "23.8", "net_cash": "10.4"},
        },
    }
    label = {
        "record_type": "label",
        "experiment_version": 2,
        "decision_id": "d1",
        "horizon": 15,
        "label_status": "valid",
        "future_p_market_up": "0.65",
        "signed_repricing": 0.05,
        "up_response_entry_result": {"net_pnl": "0.1"},
        "down_response_entry_result": {"net_pnl": "-0.2"},
    }
    records = [forecast, label, {"record_type": "market_resolved", "winner": "UP"}]
    path.write_text("\n".join(json.dumps(item) for item in records), encoding="utf-8")
    report = generate_report(path, resolve_missing=False)
    assert "side-neutral report" in report
    assert "WINNER_EDGE_ONLY: 1" in report
    assert "Jev conditioned on Chainlink TWAP side" in report


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
