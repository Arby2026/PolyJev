import asyncio
from decimal import Decimal as D

from live_market import LiveMarketState
from live_trader import DecisionScheduler, execute_choice_on_latest_state
from paper_trader import FeeSchedule, PaperTrader


def payload(value):
    return {"jev_state": {"value": value}}


def ready_state(up_ask="0.50"):
    state = LiveMarketState("up", "down")
    state.apply_event(
        {
            "event_type": "book",
            "asset_id": "up",
            "bids": [{"price": "0.49", "size": "100"}],
            "asks": [{"price": up_ask, "size": "100"}],
        }
    )
    state.apply_event(
        {
            "event_type": "book",
            "asset_id": "down",
            "bids": [{"price": "0.49", "size": "100"}],
            "asks": [{"price": "0.50", "size": "100"}],
        }
    )
    return state


def test_only_one_jev_call_in_flight():
    async def scenario():
        release = asyncio.Event()
        started = asyncio.Event()
        active = 0
        maximum = 0

        async def decide(item):
            nonlocal active, maximum
            active += 1
            maximum = max(maximum, active)
            started.set()
            await release.wait()
            active -= 1
            return item

        async def on_result(*_args):
            return None

        scheduler = DecisionScheduler(decide, on_result)
        scheduler.submit(payload(1))
        await started.wait()
        scheduler.submit(payload(2))
        release.set()
        await scheduler.wait_idle()
        assert maximum == 1
        assert scheduler.started == 2

    asyncio.run(scenario())


def test_dirty_update_runs_latest_state_after_current_call():
    async def scenario():
        release = asyncio.Event()
        started = asyncio.Event()
        seen = []

        async def decide(item):
            seen.append(item["jev_state"]["value"])
            if len(seen) == 1:
                started.set()
                await release.wait()
            return item

        async def on_result(*_args):
            return None

        scheduler = DecisionScheduler(decide, on_result)
        scheduler.submit(payload(1))
        await started.wait()
        scheduler.submit(payload(2))
        scheduler.submit(payload(3))
        release.set()
        await scheduler.wait_idle()
        assert seen == [1, 3]

    asyncio.run(scenario())


def test_duplicate_identical_state_does_not_retrigger():
    async def scenario():
        calls = 0

        async def decide(item):
            nonlocal calls
            calls += 1
            await asyncio.sleep(0)
            return item

        async def on_result(*_args):
            return None

        scheduler = DecisionScheduler(decide, on_result)
        scheduler.submit(payload(1))
        scheduler.submit(payload(1))
        await scheduler.wait_idle()
        scheduler.submit(payload(1))
        await asyncio.sleep(0)
        assert calls == 1

    asyncio.run(scenario())


def test_decision_executes_on_latest_not_stale_quotes():
    trader = PaperTrader(D("10"), FeeSchedule(False))
    stale = ready_state("0.50")
    latest = ready_state("0.80")
    assert stale.up.best_ask == D("0.50")
    result = execute_choice_on_latest_state(trader, "UP", latest, False)
    assert result.executed is True
    assert result.fills[0].price == D("0.80")
    assert trader.position.shares == D("12.5")


def test_decision_after_market_end_is_discarded():
    trader = PaperTrader(D("10"), FeeSchedule(False))
    result = execute_choice_on_latest_state(trader, "UP", ready_state(), True)
    assert result.executed is False
    assert result.block_reason == "decision discarded: market ended"
    assert trader.position.side == "FLAT"
