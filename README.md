# Jev × Polymarket — этап 3

Минимальный автономный сборщик dataset для рынков `BTC Up or Down 15m` и
`ETH Up or Down 15m`. Ручной snapshot из этапа 1 сохранён. Проект использует
только публичные REST API:

- Gamma API — точный event по вычисленному slug;
- CLOB API — реальные книги заявок outcome-токенов Up и Down;
- Binance Spot — только predictive proxy и market features.

Binance **не** считается ценой резолюции. Правила и фактический
`resolutionSource` сохраняются из Gamma; эти рынки разрешаются по Chainlink
BTC/USD или ETH/USD TWAP.

## Установка

Требуется Python 3.10+.

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Для macOS/Linux команда активации: `source .venv/bin/activate`.

Официальный OpenRouter Python SDK устанавливается из `requirements.txt`. Для
режимов с Jev ключ передаётся только через environment:

```powershell
$env:OPENROUTER_API_KEY="..."
```

Проект не читает `.env` и не сохраняет ключ в конфиге, коде или DuckDB.

## Ручной snapshot

```powershell
python run.py snapshot --asset BTC
python run.py snapshot --asset ETH
```

Для детерминированной диагностики текущего окна можно передать UTC Unix
timestamp его начала (обязательно кратный 900):

```powershell
python run.py snapshot --asset BTC --window-start 1789910100
```

Опциональное Jev enrichment выполняется только после сохранения base snapshot:

```powershell
python run.py snapshot --asset BTC --with-jev
```

Команда вычисляет slug без широкого поиска, проверяет asset и точные границы
окна из Gamma, сопоставляет `outcomes` с `clobTokenIds`, читает обе книги,
считает features и сохраняет одну строку в `data/experiment.duckdb`, таблица
`snapshots`.

## Collector

```powershell
python run.py collect
```

Collector с Jev enrichment:

```powershell
python run.py collect --with-jev
```

Collector одним процессом опрашивает локальное UTC-время и сохраняет BTC/ETH
snapshots около трёх checkpoints: `T-10`, `T-5` и `T-2`. Допустимое окно — до
15 секунд после checkpoint; пропущенные checkpoints не восстанавливаются.
Повторный запуск пропускает уже существующую пару `market_slug + checkpoint`.
После закрытия рынка outcome `UP`/`DOWN` берётся только из однозначно
разрешённых Gamma metadata и записывается во все snapshots рынка.

Остановить collector можно через `Ctrl+C`.

## Status

```powershell
python run.py status
```

Команда читает локальную DuckDB без сетевых запросов и показывает количество
рынков, snapshots, checkpoints, resolved/unresolved markets и последний snapshot.

## Stage 4 analysis

```powershell
python analyze.py
```

Stage 4 compares Market, `P_simple`, Jev Blind and Jev Meta using resolved
checkpoint observations.

## Realtime Jev Polymarket Paper Trader

Задайте `OPENROUTER_API_KEY` в environment и запустите trader для текущего
15-минутного рынка:

```powershell
python live_trader.py --asset BTC
```

Дополнительные варианты:

```powershell
python live_trader.py --asset ETH
python live_trader.py --asset BTC --notional 10
```

Trader использует только Polymarket state из публичного realtime WebSocket,
держит не более одного Jev inference в полёте и трактует `UP`/`DOWN`/`FLAT` как
target position. Комиссии читаются из текущего рынка; paper taker execution
покупает по asks и продаёт по bids, а settlement платит 1/0 без taker fee.
Реальные ордера и денежные средства не используются.

## Forecast Experiment

Отдельный realtime-эксперимент не торгует и не управляет позициями. Он строит
position-independent Jev forecasts по live Polymarket CLOB и Chainlink raw +
TWAP60 из Polymarket RTDS. Jev отвечает одним Noul `resolve_up`; DOWN всегда
вычисляется как бинарное дополнение. Input и post-inference response состояния
логируются раздельно, а детерминированные 15/30/60/120-секундные labels измеряют
симметричный market repricing и net PnL обеих сторон по executable bids. Report
сравнивает Jev с normalized market и Chainlink baselines без trading controller.
JSONL V2 помечен `experiment_version: 2`; старые forecast JSONL также читаются.
Лимит расходов Jev по умолчанию — `$0.15` на одну полную рыночную сессию.

```powershell
python forecast_experiment.py --asset BTC --max-jev-cost 0.15
```

Программа сама подписывается на Chainlink до следующего 15-минутного окна,
проводит один полный market, получает resolution и печатает report. Повторный
отчёт по сохранённой сессии:

```powershell
python forecast_report.py data/forecast_btc_<window_start>.jsonl
```

`P_simple` — zero-drift probability baseline, рассчитанная по Binance proxy и
realized volatility; она **не** является settlement probability source.

`P_jev_blind` строится без Polymarket prices, quotes и `P_simple`.
`P_jev_meta` видит тот же underlying state, а также `P_simple`, Polymarket UP
midpoint и текущие UP/DOWN bid/ask. Оба значения — отдельные Noul-вызовы
нативного OpenRouter Decisions API через официальный Python SDK и модель
`~typesafe/jev-latest`.

`return_1m` и `return_5m` приближены по доступным 1-minute candles и не обещают
sub-minute точность. Returns и `realized_vol_5m`/`realized_vol_15m` хранятся как
доли (не проценты). Волатильность — population standard deviation минутных
лог-доходностей без annualization. `distance_from_start_bps` хранится в bps.

## Тесты

```powershell
python -m pytest -q
```

Тесты покрывают логику этапов 1–2: окна и slug, mapping outcomes, лучшие цены,
features, checkpoints, `P_simple`, duplicate detection, parsing resolution,
изоляцию Blind/Meta state и обновление Jev-полей в DuckDB.

## Использованная официальная документация

- [OpenRouter Python SDK](https://openrouter.ai/docs/client-sdks/python/overview)
- [TypeSafe models on OpenRouter](https://openrouter.ai/typesafe)
- [Jev Latest](https://openrouter.ai/~typesafe/jev-latest)
- [Jev compiler](https://openrouter.ai/labs/jev/compile)
- [Discover Markets](https://docs.polymarket.com/market-data/discover-markets)
- [Prices and Order Books](https://docs.polymarket.com/market-data/prices-order-books)
- [Chainlink TWAP Prices](https://docs.polymarket.com/market-data/chainlink-twap)
