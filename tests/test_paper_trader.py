from decimal import Decimal as D

import pytest

from paper_trader import (
    FeeSchedule,
    Level,
    PaperTrader,
    calculate_fee,
    quote_buy,
    quote_sell,
)


ENABLED = FeeSchedule(True, D("0.07"), D("1"), True, D("0.20"))
DISABLED = FeeSchedule(False)


def books(up_ask="0.50", down_ask="0.50", bid="0.49", size="100"):
    return {
        "UP": {
            "bids": [Level(D(bid), D(size))],
            "asks": [Level(D(up_ask), D(size))],
        },
        "DOWN": {
            "bids": [Level(D(bid), D(size))],
            "asks": [Level(D(down_ask), D(size))],
        },
    }


@pytest.mark.parametrize(
    ("price", "expected"), [("0.50", "1.75000"), ("0.10", "0.63000")]
)
def test_fee_formula(price, expected):
    assert calculate_fee(D("100"), D(price), ENABLED) == D(expected)


def test_fee_disabled():
    assert calculate_fee(D("100"), D("0.50"), DISABLED) == D("0")


def test_buy_cash_flow_includes_fee():
    quote = quote_buy([Level(D("0.50"), D("100"))], D("10"), ENABLED)
    assert quote.gross_value == D("10")
    assert quote.fee == D("0.35000")
    assert quote.net_cash == D("10.35000")


def test_sell_proceeds_subtract_fee():
    quote = quote_sell([Level(D("0.50"), D("100"))], D("20"), ENABLED)
    assert quote.gross_value == D("10")
    assert quote.fee == D("0.35000")
    assert quote.net_cash == D("9.65000")


def test_spread_is_not_charged_twice():
    trader = PaperTrader(D("10"), DISABLED)
    market = books(up_ask="0.60", bid="0.40")
    trader.execute_target("UP", market)
    closed = trader.execute_target("FLAT", market)
    assert closed.realized_pnl_delta == pytest.approx(D("-3.333333333333333333333333333"))


@pytest.mark.parametrize("target", ["UP", "DOWN"])
def test_flat_opens_target(target):
    trader = PaperTrader(D("10"), DISABLED)
    result = trader.execute_target(target, books())
    assert result.executed is True
    assert result.kind == "open"
    assert trader.position.side == target


@pytest.mark.parametrize("side", ["UP", "DOWN"])
def test_same_target_does_not_trade(side):
    trader = PaperTrader(D("10"), DISABLED)
    trader.execute_target(side, books())
    result = trader.execute_target(side, books())
    assert result.executed is False
    assert result.block_reason == "target unchanged"
    assert trader.trades == 1


@pytest.mark.parametrize("side", ["UP", "DOWN"])
def test_flat_target_closes_at_executable_bid(side):
    trader = PaperTrader(D("10"), DISABLED)
    trader.execute_target(side, books(up_ask="0.50", down_ask="0.50", bid="0.40"))
    result = trader.execute_target("FLAT", books(bid="0.40"))
    assert result.executed is True
    assert result.fills[0].action == "SELL"
    assert result.fills[0].price == D("0.40")
    assert trader.position.side == "FLAT"


@pytest.mark.parametrize(("start", "target"), [("UP", "DOWN"), ("DOWN", "UP")])
def test_flip(start, target):
    trader = PaperTrader(D("10"), DISABLED)
    trader.execute_target(start, books())
    result = trader.execute_target(target, books())
    assert result.executed is True
    assert result.kind == "flip"
    assert [fill.action for fill in result.fills] == ["SELL", "BUY"]
    assert trader.position.side == target


def test_insufficient_liquidity_blocks_entry():
    trader = PaperTrader(D("10"), DISABLED)
    result = trader.execute_target("UP", books(size="1"))
    assert result.executed is False
    assert result.block_reason == "insufficient executable liquidity"
    assert trader.position.side == "FLAT"


def test_insufficient_liquidity_blocks_whole_flip():
    trader = PaperTrader(D("10"), DISABLED)
    trader.execute_target("UP", books())
    before = trader.position.snapshot()
    market = books()
    market["DOWN"]["asks"] = [Level(D("0.50"), D("1"))]
    result = trader.execute_target("DOWN", market)
    assert result.executed is False
    assert result.block_reason == "flip blocked: insufficient executable liquidity"
    assert trader.position.snapshot() == before
    assert trader.realized_trading_pnl == D("0")


@pytest.mark.parametrize(("winner", "expected"), [("UP", "10"), ("DOWN", "-10")])
def test_settlement_pays_one_or_zero_without_fee(winner, expected):
    trader = PaperTrader(D("10"), DISABLED)
    trader.execute_target("UP", books())
    fees_before = trader.total_fees
    pnl = trader.settle(winner)
    assert pnl == D(expected)
    assert trader.total_fees == fees_before
    assert trader.position.side == "FLAT"


def test_unrealized_pnl_uses_executable_exit_and_exit_fee():
    trader = PaperTrader(D("10"), ENABLED)
    trader.execute_target("UP", books(up_ask="0.50", bid="0.40"))
    mark = trader.mark_to_market(books(bid="0.40"))
    assert mark["exit_fee"] == D("0.33600")
    assert mark["liquidation_value_net"] == D("7.66400")
    assert mark["unrealized_pnl"] == D("-2.68600")
