from types import SimpleNamespace

import duckdb
import pytest

from jev import (
    build_blind_state,
    build_meta_state,
    call_jev,
    parse_choice_response,
    validate_probability,
)
from run import _require_jev_api_key
from storage import initialize_database, update_jev_results


SNAPSHOT = {
    "asset": "BTC",
    "rules": "Resolve UP when the Chainlink TWAP finishes at or above its start.",
    "resolution_source": "https://data.chain.link/streams/btc-usd-twap-60s-streams",
    "time_remaining_sec": 300.0,
    "proxy_start_price": 80_000.0,
    "proxy_current_price": 80_100.0,
    "distance_from_start_bps": 12.5,
    "return_1m": 0.001,
    "return_5m": 0.002,
    "realized_vol_5m": 0.003,
    "realized_vol_15m": 0.004,
    "p_simple": 0.72,
    "p_market": 0.68,
    "up_best_bid": 0.67,
    "up_best_ask": 0.69,
    "down_best_bid": 0.31,
    "down_best_ask": 0.33,
    "outcome": None,
}


class FakeDecisions:
    def __init__(self, response):
        self.response = response
        self.request = None

    def create(self, **kwargs):
        self.request = kwargs
        return self.response


class FakeClient:
    def __init__(self, response):
        self.decisions = FakeDecisions(response)
        self.alpha = SimpleNamespace(decisions=self.decisions)


def test_blind_state_excludes_polymarket_and_baseline_fields():
    state = build_blind_state(SNAPSHOT)
    forbidden = {
        "p_market",
        "p_simple",
        "up_best_bid",
        "up_best_ask",
        "down_best_bid",
        "down_best_ask",
        "outcome",
    }
    assert forbidden.isdisjoint(state)
    assert state["asset"] == "BTC"
    assert "predictive proxy only" in state["proxy_notice"]


def test_meta_state_includes_market_and_baseline_fields():
    state = build_meta_state(SNAPSHOT)
    required = {
        "p_market",
        "p_simple",
        "up_best_bid",
        "up_best_ask",
        "down_best_bid",
        "down_best_ask",
    }
    assert required.issubset(state)
    assert state["p_market"] == 0.68
    assert state["p_simple"] == 0.72


@pytest.mark.parametrize("value", [0, 0.5, 1])
def test_probability_validation_accepts_closed_interval(value):
    assert validate_probability(value) == float(value)


@pytest.mark.parametrize("value", [-0.01, 1.01, float("inf")])
def test_probability_validation_rejects_outside_interval(value):
    with pytest.raises(RuntimeError, match="between 0 and 1"):
        validate_probability(value)


def test_missing_openrouter_key_is_understandable(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY is not set"):
        _require_jev_api_key()


def test_decisions_noul_answer_is_parsed():
    response = SimpleNamespace(
        answers={"resolves_up": SimpleNamespace(type="noul", noul=0.64)},
        model="typesafe/jev-1.13",
        usage=SimpleNamespace(input_tokens=123),
    )
    client = FakeClient(response)

    result = call_jev(client, "~typesafe/jev-latest", {"asset": "BTC"}, "Decide")

    assert result["probability"] == 0.64
    assert result["model"] == "typesafe/jev-1.13"
    assert result["input_tokens"] == 123
    assert client.decisions.request == {
        "model": "~typesafe/jev-latest",
        "state": {"asset": "BTC"},
        "questions": {
            "resolves_up": {"type": "noul", "instructions": "Decide"}
        },
    }


@pytest.mark.parametrize(
    "answers",
    [
        {},
        {"resolves_up": SimpleNamespace(type="noul")},
    ],
)
def test_malformed_or_missing_resolves_up_raises_runtime_error(answers):
    response = SimpleNamespace(answers=answers, model=None, usage=None)
    with pytest.raises(RuntimeError, match="missing resolves_up Noul"):
        call_jev(FakeClient(response), "~typesafe/jev-latest", {}, "Decide")


@pytest.mark.parametrize("choice", ["UP", "DOWN", "FLAT"])
def test_target_choice_parsing(choice):
    response = SimpleNamespace(
        answers={
            "target_position": SimpleNamespace(
                choice=choice,
                probabilities={"UP": 0.6, "DOWN": 0.2, "FLAT": 0.2},
            )
        },
        model="typesafe/jev-1.13",
        usage=SimpleNamespace(input_tokens=77),
    )
    result = parse_choice_response(
        response, "target_position", ("UP", "DOWN", "FLAT")
    )
    assert result["choice"] == choice
    assert result["probabilities"] == {"UP": 0.6, "DOWN": 0.2, "FLAT": 0.2}
    assert result["input_tokens"] == 77


def test_schema_upgrade_and_jev_update(tmp_path):
    database = str(tmp_path / "legacy.duckdb")
    snapshot_id = "00000000-0000-0000-0000-000000000123"
    with duckdb.connect(database) as connection:
        connection.execute(
            """
            CREATE TABLE snapshots (
                snapshot_id UUID PRIMARY KEY,
                market_slug VARCHAR,
                checkpoint VARCHAR
            )
            """
        )
        connection.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?)",
            [snapshot_id, "btc-updown-15m-test", "T-5"],
        )

    initialize_database(database)
    with duckdb.connect(database) as connection:
        columns = {row[0] for row in connection.execute("DESCRIBE snapshots").fetchall()}
    expected_columns = {
        "p_jev_blind",
        "p_jev_meta",
        "jev_model",
        "jev_blind_latency_ms",
        "jev_meta_latency_ms",
        "jev_blind_input_tokens",
        "jev_meta_input_tokens",
    }
    assert expected_columns.issubset(columns)

    update_jev_results(
        database,
        snapshot_id,
        p_jev_blind=0.61,
        p_jev_meta=0.73,
        jev_model="jev-test",
        jev_blind_latency_ms=101.5,
        jev_meta_latency_ms=112.5,
        jev_blind_input_tokens=120,
        jev_meta_input_tokens=150,
    )
    with duckdb.connect(database) as connection:
        result = connection.execute(
            """
            SELECT p_jev_blind, p_jev_meta, jev_model,
                   jev_blind_latency_ms, jev_meta_latency_ms,
                   jev_blind_input_tokens, jev_meta_input_tokens
            FROM snapshots WHERE snapshot_id = ?
            """,
            [snapshot_id],
        ).fetchone()
    assert result == (0.61, 0.73, "jev-test", 101.5, 112.5, 120, 150)
