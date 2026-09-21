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

Задайте `OPENROUTER_API_KEY` в environment или скопируйте `.env.example`
в локальный `.env` и заполните ключ. Оба realtime-скрипта читают `.env` из
текущего каталога, только если ключ не задан в окружении. `.env` исключён
из Git; ключ не выводится в логи. Если ключ не задан ни одним способом,
программа печатает инструкцию настройки и завершается с кодом 1.
Запустите trader для текущего
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
TWAP60 из Polymarket RTDS. Jev отвечает одним Noul Q1 `p_final_up`; DOWN всегда
вычисляется как бинарное дополнение. Input и post-inference response состояния
логируются раздельно, а детерминированные 15/30/60/120-секундные labels измеряют
симметричный market repricing и net PnL обеих сторон по executable bids. Report
сравнивает Jev с normalized market и Chainlink baselines без trading controller.
JSONL V3 помечен `experiment_version: 3`; V2 и старые forecast JSONL также читаются.
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

V3 передаёт Jev человекочитаемый Meta state внутри объекта `{"meta": "..."}`.
State пересобирается непосредственно перед запросом: `Time left`, `Fees`,
`Flow 60s`, `Book`, `Wall`, `Chainlink vol 60s`. Это один прогноз финального
резолва, а не отдельные вопросы о скальпе. Старые Blind/Meta в collector не меняются.

- `time_left_sec = max(0, int(window_start + 900 - now))`;
  `distance_from_start_bps` — Chainlink raw относительно наблюдаемого opening.
- `flow_60s` — RTDS `activity/trades` с локальной проверкой slug и token ID.
  Серверные фильтры `market_slug` и `event_slug` на проверке 2026-09-21
  не отдавали сделки активного рынка; публичная подписка без фильтра их
  отдаёт. Чужие рынки отбрасываются до помещения в deque.
  Только BUY: `up_count`, `down_count`, `net = up_count - down_count`,
  `avg_size` и средние по сторонам в USD (`price * size`),
  `imbalance_usd = up_usd - down_usd`. SELL не превращается в BUY другой стороны.
  Старые события и дубликаты отфильтровываются; без полной подписки в течение
  60 секунд flow помечается `warming_up`/`unavailable`.
- `mid` — ненормализованный UP midpoint. `spread_bps = (ask-bid)/mid * 10000`
  для UP; в `book` также есть DOWN. 1% = 100 bps.
  `depth_ratio` — отношение USD-глубины UP/DOWN по пяти лучшим bid и ask.
- Стенка: размер лучшего уровня в shares строго больше 3 средних размеров
  следующих пяти уровней и USD-размер строго больше
  `max(1500, 0.25 * total_depth_usd)`. Здесь total depth — все bid и ask
  данного outcome; проверяются BID/ASK для UP/DOWN. Менее шести уровней
  недостаточно для обнаружения стенки.
- `vol_60s` — population std последовательных log returns Chainlink raw
  за последние 60 секунд, умноженная на 100, без annualization. История
  ограничивается временем, а не 100 тиками. `choppy`: минимум две смены
  направления и не менее половины переходов ненулевого направления;
  иначе `trend`, для неизменной цены `flat`. Менее трёх точек даёт `null`
  и `insufficient_data`; длительность покрытия также логируется.
- `fee_schedule_bps` — коэффициент fee schedule из того же parser, что
  использует `live_trader.py`. `fee_effective_up_bps` и
  `fee_effective_down_bps` — оценки effective taker fees для $10 по mid
  соответствующей стороны: `schedule_rate * (1 - mid) * 10000`.
  Например, 7% schedule и mid 0.50 дают 350 bps effective. Отчёт показывает
  средние effective UP/DOWN, а не коэффициент schedule. Комиссии $10 BUY
  по исполнимым asks сохранены отдельно в `effective_buy_fees_bps`.
  Старое неоднозначное поле `fees_bps` в новые записи не добавляется.

До вызова Jev `mid < 0.10` или `mid > 0.90` даёт `forecast_skip` с
`skipped: true`, `skip_reason`, `state_text` и всеми признаками, без Jev-полей.
Равенство 0.10/0.90 допускается. Исчерпание бюджета тоже логируется как skip.
Кандидаты поступают не чаще раза в секунду; во время запроса сохраняется
последний кандидат. Поэтому skipped % — доля записанных кандидатов, не WS ticks.
Для выполненного прогноза `skipped: false`, `skip_reason: null`,
`p_jev_up` сохранён как совместимый alias `p_final_up`.

Report показывает Pearson Q1 относительно входных `distance_from_start_bps`
и `time_left_sec`, с порогом прогресса **|r| < 0.75**, включая отрицательную
зависимость. FAIL корреляции не означает ошибку кода: нужен ещё один рынок
для статистики. Изначальная исследовательская цель — |r| < 0.6.
Он не подменяет входные цены ценами после ответа и не включает скипы.
Для V2 используются pre-inference поля `input_state`. Недостаток данных
или постоянная переменная даёт `n/a`, а не успешную проверку. Корреляции и
доля скипов доступны до резолва; распределение flow для верных/неверных
прогнозов появляется после официального резолва, только для полного flow.
Один рынок не подтверждает общее снижение корреляции или рост win rate.

Проверка V3 на полном следующем 15-минутном рынке (ключ из environment или `.env`):

```powershell
python -m pytest -q
python forecast_experiment.py --asset BTC --max-jev-cost 0.05
python forecast_report.py data/forecast_btc_<window_start>.jsonl
```

Для отчётов по нескольким сессиям можно передать несколько путей либо
`python forecast_report.py "data/forecast_btc_*.jsonl"` (включая PowerShell).

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
