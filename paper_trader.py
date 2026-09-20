"""Deterministic fee-aware taker execution for the realtime paper trader."""

from __future__ import annotations

from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Literal, Mapping, Sequence


Side = Literal["UP", "DOWN", "FLAT"]
Action = Literal["BUY", "SELL"]
ZERO = Decimal("0")
ONE = Decimal("1")
FEE_QUANTUM = Decimal("0.00001")


@dataclass(frozen=True)
class FeeSchedule:
    enabled: bool
    rate: Decimal = ZERO
    exponent: Decimal = ONE
    taker_only: bool = True
    rebate_rate: Decimal = ZERO

    def validate(self) -> None:
        if self.enabled and (
            self.rate < ZERO or self.exponent != ONE or not self.taker_only
        ):
            raise ValueError("unsupported fee schedule for paper simulation")


@dataclass(frozen=True)
class Level:
    price: Decimal
    size: Decimal


@dataclass(frozen=True)
class Fill:
    action: Action
    price: Decimal
    shares: Decimal
    gross_value: Decimal
    fee: Decimal

    def as_dict(self) -> dict[str, str]:
        return {
            "action": self.action,
            "price": str(self.price),
            "shares": str(self.shares),
            "gross_value": str(self.gross_value),
            "fee": str(self.fee),
        }


@dataclass(frozen=True)
class ExecutionQuote:
    action: Action
    requested: Decimal
    shares: Decimal
    gross_value: Decimal
    fee: Decimal
    net_cash: Decimal
    vwap: Decimal | None
    fully_executable: bool
    fills: tuple[Fill, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "action": self.action,
            "requested": str(self.requested),
            "shares": str(self.shares),
            "gross_value": str(self.gross_value),
            "fee": str(self.fee),
            "net_cash": str(self.net_cash),
            "vwap": None if self.vwap is None else str(self.vwap),
            "fully_executable": self.fully_executable,
            "fills": [fill.as_dict() for fill in self.fills],
        }


@dataclass
class Position:
    side: Side = "FLAT"
    shares: Decimal = ZERO
    entry_gross_value: Decimal = ZERO
    entry_fee: Decimal = ZERO
    total_entry_cost: Decimal = ZERO

    def snapshot(self) -> dict[str, str]:
        return {
            "side": self.side,
            "shares": str(self.shares),
            "entry_gross_value": str(self.entry_gross_value),
            "entry_fee": str(self.entry_fee),
            "total_entry_cost": str(self.total_entry_cost),
        }


@dataclass(frozen=True)
class TransitionResult:
    target: Side
    position_before: dict[str, str]
    position_after: dict[str, str]
    executed: bool
    block_reason: str | None
    fills: tuple[Fill, ...]
    fees_paid: Decimal
    realized_pnl_delta: Decimal
    kind: Literal["none", "open", "close", "flip"]


def calculate_fee(shares: Decimal, price: Decimal, schedule: FeeSchedule) -> Decimal:
    schedule.validate()
    if not schedule.enabled:
        return ZERO
    raw = shares * schedule.rate * price * (ONE - price)
    return raw.quantize(FEE_QUANTUM, rounding=ROUND_HALF_UP)


def quote_buy(
    asks: Sequence[Level], notional: Decimal, schedule: FeeSchedule
) -> ExecutionQuote:
    remaining = notional
    fills: list[Fill] = []
    for level in sorted(asks, key=lambda item: item.price):
        if remaining <= ZERO:
            break
        if level.price <= ZERO or level.size <= ZERO:
            continue
        level_gross = min(remaining, level.price * level.size)
        shares = level_gross / level.price
        fee = calculate_fee(shares, level.price, schedule)
        fills.append(Fill("BUY", level.price, shares, level_gross, fee))
        remaining -= level_gross
    return _finish_quote("BUY", notional, fills, remaining <= ZERO)


def quote_sell(
    bids: Sequence[Level], shares: Decimal, schedule: FeeSchedule
) -> ExecutionQuote:
    remaining = shares
    fills: list[Fill] = []
    for level in sorted(bids, key=lambda item: item.price, reverse=True):
        if remaining <= ZERO:
            break
        if level.price <= ZERO or level.size <= ZERO:
            continue
        level_shares = min(remaining, level.size)
        gross = level_shares * level.price
        fee = calculate_fee(level_shares, level.price, schedule)
        fills.append(Fill("SELL", level.price, level_shares, gross, fee))
        remaining -= level_shares
    return _finish_quote("SELL", shares, fills, remaining <= ZERO)


def _finish_quote(
    action: Action, requested: Decimal, fills: Sequence[Fill], complete: bool
) -> ExecutionQuote:
    shares = sum((fill.shares for fill in fills), ZERO)
    gross = sum((fill.gross_value for fill in fills), ZERO)
    fee = sum((fill.fee for fill in fills), ZERO)
    net_cash = gross + fee if action == "BUY" else gross - fee
    return ExecutionQuote(
        action=action,
        requested=requested,
        shares=shares,
        gross_value=gross,
        fee=fee,
        net_cash=net_cash,
        vwap=gross / shares if shares else None,
        fully_executable=complete,
        fills=tuple(fills),
    )


class PaperTrader:
    def __init__(self, notional: Decimal, fee_schedule: FeeSchedule):
        fee_schedule.validate()
        self.notional = notional
        self.fee_schedule = fee_schedule
        self.position = Position()
        self.realized_trading_pnl = ZERO
        self.settlement_pnl = ZERO
        self.total_fees = ZERO
        self.trades = 0
        self.opens = 0
        self.closes = 0
        self.flips = 0
        self.trade_history: list[dict[str, Any]] = []

    def preview_transitions(
        self, books: Mapping[str, Mapping[str, Sequence[Level]]]
    ) -> dict[str, dict[str, Any]]:
        return {
            target: self._preview_target(target, books)
            for target in ("UP", "DOWN", "FLAT")
        }

    def _preview_target(
        self, target: Side, books: Mapping[str, Mapping[str, Sequence[Level]]]
    ) -> dict[str, Any]:
        current = self.position.side
        if target == current:
            return {"executable": True, "transition_cost": "0", "action": "hold"}
        close_quote = None
        open_quote = None
        if current != "FLAT":
            close_quote = quote_sell(
                books[current]["bids"], self.position.shares, self.fee_schedule
            )
        if target != "FLAT":
            open_quote = quote_buy(
                books[target]["asks"], self.notional, self.fee_schedule
            )
        executable = bool(
            (close_quote is None or close_quote.fully_executable)
            and (open_quote is None or open_quote.fully_executable)
        )
        result: dict[str, Any] = {
            "executable": executable,
            "action": "flip" if close_quote and open_quote else "close" if close_quote else "open",
            "close": close_quote.as_dict() if close_quote else None,
            "open": open_quote.as_dict() if open_quote else None,
        }
        if close_quote:
            result["realized_pnl_if_executed"] = str(
                close_quote.net_cash - self.position.total_entry_cost
            )
        result["total_transition_fees"] = str(
            (close_quote.fee if close_quote else ZERO)
            + (open_quote.fee if open_quote else ZERO)
        )
        return result

    def execute_target(
        self, target: Side, books: Mapping[str, Mapping[str, Sequence[Level]]]
    ) -> TransitionResult:
        if target not in ("UP", "DOWN", "FLAT"):
            raise ValueError(f"invalid target position: {target}")
        before = self.position.snapshot()
        current = self.position.side
        if target == current:
            return TransitionResult(
                target, before, before.copy(), False, "target unchanged", (), ZERO, ZERO, "none"
            )

        close_quote = (
            quote_sell(books[current]["bids"], self.position.shares, self.fee_schedule)
            if current != "FLAT"
            else None
        )
        open_quote = (
            quote_buy(books[target]["asks"], self.notional, self.fee_schedule)
            if target != "FLAT"
            else None
        )
        if close_quote is not None and not close_quote.fully_executable:
            reason = (
                "flip blocked: insufficient executable liquidity"
                if open_quote is not None
                else "insufficient executable liquidity"
            )
            return TransitionResult(
                target, before, before.copy(), False, reason, (), ZERO, ZERO, "none"
            )
        if open_quote is not None and not open_quote.fully_executable:
            reason = (
                "flip blocked: insufficient executable liquidity"
                if close_quote is not None
                else "insufficient executable liquidity"
            )
            return TransitionResult(
                target, before, before.copy(), False, reason, (), ZERO, ZERO, "none"
            )

        fills: tuple[Fill, ...] = ()
        fees = ZERO
        realized = ZERO
        kind: Literal["open", "close", "flip"]
        if close_quote is not None:
            fills += close_quote.fills
            fees += close_quote.fee
            realized = close_quote.net_cash - self.position.total_entry_cost
            self.realized_trading_pnl += realized
            self.position = Position()
        if open_quote is not None:
            fills += open_quote.fills
            fees += open_quote.fee
            self.position = Position(
                side=target,
                shares=open_quote.shares,
                entry_gross_value=open_quote.gross_value,
                entry_fee=open_quote.fee,
                total_entry_cost=open_quote.net_cash,
            )
        if close_quote is not None and open_quote is not None:
            kind = "flip"
            self.flips += 1
        elif close_quote is not None:
            kind = "close"
            self.closes += 1
        else:
            kind = "open"
            self.opens += 1
        self.trades += 1
        self.total_fees += fees
        result = TransitionResult(
            target,
            before,
            self.position.snapshot(),
            True,
            None,
            fills,
            fees,
            realized,
            kind,
        )
        self.trade_history.append(
            {
                "kind": kind,
                "from": before["side"],
                "to": target,
                "fees": str(fees),
                "realized_pnl": str(realized),
                "fills": [fill.as_dict() for fill in fills],
            }
        )
        return result

    def mark_to_market(
        self, books: Mapping[str, Mapping[str, Sequence[Level]]]
    ) -> dict[str, Decimal | None]:
        if self.position.side == "FLAT":
            return {"liquidation_value_net": ZERO, "exit_fee": ZERO, "unrealized_pnl": ZERO}
        quote = quote_sell(
            books[self.position.side]["bids"], self.position.shares, self.fee_schedule
        )
        if not quote.fully_executable:
            return {"liquidation_value_net": None, "exit_fee": None, "unrealized_pnl": None}
        return {
            "liquidation_value_net": quote.net_cash,
            "exit_fee": quote.fee,
            "unrealized_pnl": quote.net_cash - self.position.total_entry_cost,
        }

    def settle(self, winner: Literal["UP", "DOWN"]) -> Decimal:
        if self.position.side == "FLAT":
            return ZERO
        proceeds = self.position.shares if self.position.side == winner else ZERO
        pnl = proceeds - self.position.total_entry_cost
        self.settlement_pnl += pnl
        self.position = Position()
        return pnl

    @property
    def total_pnl(self) -> Decimal:
        return self.realized_trading_pnl + self.settlement_pnl
