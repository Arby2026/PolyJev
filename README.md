# Jev × Polymarket — этап 1

Минимальный сборщик одного snapshot для текущего рынка `BTC Up or Down 15m`
или `ETH Up or Down 15m`. Он использует только публичные REST API:

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

## Запуск

```powershell
python run.py snapshot --asset BTC
python run.py snapshot --asset ETH
```

Для детерминированной диагностики текущего окна можно передать UTC Unix
timestamp его начала (обязательно кратный 900):

```powershell
python run.py snapshot --asset BTC --window-start 1789910100
```

Команда вычисляет slug без широкого поиска, проверяет asset и точные границы
окна из Gamma, сопоставляет `outcomes` с `clobTokenIds`, читает обе книги,
считает features и сохраняет одну строку в `data/experiment.duckdb`, таблица
`snapshots`.

`return_1m`, `return_5m`, `realized_vol_5m` и `realized_vol_15m` хранятся как
доли (не проценты). Волатильность — population standard deviation минутных
лог-доходностей, без annualization. `distance_from_start_bps` хранится в bps.

## Тесты

```powershell
python -m pytest -q
```

Тесты покрывают округление окна, slug, mapping outcome → token, извлечение
лучших цен без предположения о сортировке и расчёт features на synthetic data.

## Использованная официальная документация

- [Discover Markets](https://docs.polymarket.com/market-data/discover-markets)
- [Prices and Order Books](https://docs.polymarket.com/market-data/prices-order-books)
- [Chainlink TWAP Prices](https://docs.polymarket.com/market-data/chainlink-twap)

