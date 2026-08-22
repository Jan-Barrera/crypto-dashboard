from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from config import (
    asset_name,
    format_compact_usd,
    format_usd,
    load_crypto_pairs,
    to_binance_symbol,
    to_display_pair,
)
from db.binance import BinanceAPIError, binance_get

logger = logging.getLogger(__name__)

SCRAPE_INTERVAL_SECONDS = 5 * 60
_TICKER_CACHE: tuple[float, list[dict[str, Any]]] | None = None


def _fallback_tickers() -> list[dict[str, Any]]:
    return [
        {
            "symbol": "BTC/USDT",
            "binance_symbol": "BTCUSDT",
            "name": "Bitcoin",
            "price": 0.0,
            "change": 0.0,
            "percent": 0.0,
            "high": 0.0,
            "low": 0.0,
            "volume": 0.0,
            "quote_volume": 0.0,
            "trades": 0,
            "up": True,
        }
    ]


def _parse_ticker(row: dict[str, Any]) -> dict[str, Any]:
    percent = float(row.get("priceChangePercent") or 0)
    return {
        "symbol": to_display_pair(row["symbol"]),
        "binance_symbol": row["symbol"],
        "name": asset_name(row["symbol"]),
        "price": float(row.get("lastPrice") or 0),
        "change": float(row.get("priceChange") or 0),
        "percent": percent,
        "high": float(row.get("highPrice") or 0),
        "low": float(row.get("lowPrice") or 0),
        "volume": float(row.get("volume") or 0),
        "quote_volume": float(row.get("quoteVolume") or 0),
        "trades": int(float(row.get("count") or 0)),
        "up": percent >= 0,
    }


def _fetch_tickers() -> list[dict[str, Any]]:
    pairs = load_crypto_pairs()
    symbols = [to_binance_symbol(pair) for pair in pairs]
    wanted = set(symbols)
    try:
        payload = binance_get(
            "/api/v3/ticker/24hr",
            {"symbols": json.dumps(symbols, separators=(",", ":"))},
        )
    except BinanceAPIError:
        payload = binance_get("/api/v3/ticker/24hr")
    if isinstance(payload, dict):
        payload = [payload]
    if not isinstance(payload, list) or not payload:
        raise ValueError("Binance 24hr ticker returned no rows")

    parsed = [
        _parse_ticker(row)
        for row in payload
        if isinstance(row, dict) and row.get("symbol") in wanted
    ]
    by_symbol = {item["binance_symbol"]: item for item in parsed}
    tickers: list[dict[str, Any]] = []
    for pair in pairs:
        ticker = by_symbol.get(to_binance_symbol(pair))
        if ticker:
            tickers.append(ticker)
        else:
            logger.warning("No Binance 24hr ticker for %s", pair)
    if not tickers:
        raise ValueError("No matching data.")
    return tickers


def get_tickers() -> list[dict[str, Any]]:
    global _TICKER_CACHE

    now = time.time()
    if _TICKER_CACHE and now - _TICKER_CACHE[0] < SCRAPE_INTERVAL_SECONDS:
        return _TICKER_CACHE[1]

    try:
        tickers = _fetch_tickers()
    except (BinanceAPIError, ValueError, KeyError, TypeError) as exc:
        logger.warning("Binance 24hr ticker fetch failed: %s", exc)
        if _TICKER_CACHE:
            return _TICKER_CACHE[1]
        tickers = _fallback_tickers()

    _TICKER_CACHE = (now, tickers)
    return tickers


def get_ticker_map() -> dict[str, dict[str, Any]]:
    return {ticker["symbol"]: ticker for ticker in get_tickers()}


def get_indices() -> list[dict[str, Any]]:
    cards: list[dict[str, Any]] = []
    for ticker in get_tickers()[:4]:
        sign = "+" if ticker["change"] >= 0 else ""
        percent_sign = "+" if ticker["percent"] >= 0 else ""
        cards.append(
            {
                "name": ticker["name"].upper(),
                "value": format_usd(ticker["price"]),
                "change": (
                    f"{sign}{format_usd(ticker['change'])} "
                    f"({percent_sign}{ticker['percent']:.2f}%)"
                ),
                "up": ticker["up"],
                "value_b": format_compact_usd(ticker["quote_volume"]),
                "vol": format_usd(ticker["high"]),
                "trades": format_usd(ticker["low"]),
            }
        )
    return cards


def get_indices_updated_at() -> datetime | None:
    if _TICKER_CACHE is None:
        return None
    return datetime.fromtimestamp(_TICKER_CACHE[0], tz=ZoneInfo("Asia/Manila"))


def indices_needs_refresh() -> bool:
    if _TICKER_CACHE is None:
        return True
    return time.time() - _TICKER_CACHE[0] >= SCRAPE_INTERVAL_SECONDS


def get_market_breadth() -> dict[str, int]:
    tickers = get_tickers()
    advancers = sum(1 for ticker in tickers if ticker["percent"] > 0)
    decliners = sum(1 for ticker in tickers if ticker["percent"] < 0)
    unchanged = len(tickers) - advancers - decliners
    return {
        "advancers": advancers,
        "decliners": decliners,
        "unchanged": unchanged,
        "total": len(tickers),
    }


def get_watchlist_data() -> pd.DataFrame:
    rows = []
    for index, ticker in enumerate(get_tickers(), start=1):
        percent = ticker["percent"]
        sign = "+" if percent >= 0 else ""
        rows.append(
            {
                "#": str(index),
                "Symbol": ticker["symbol"],
                "Name": ticker["name"],
                "Last Price": format_usd(ticker["price"]),
                "% Change": f"{sign}{percent:.2f}%",
                "24h High": format_usd(ticker["high"]),
                "24h Low": format_usd(ticker["low"]),
                "Volume": format_compact_usd(ticker["quote_volume"]),
            }
        )
    return pd.DataFrame(rows)


def get_market_news() -> list[tuple[str, str, str]]:
    tickers = sorted(get_tickers(), key=lambda item: item["percent"], reverse=True)
    movers: list[tuple[str, str, str]] = []
    gainers = [ticker for ticker in tickers if ticker["percent"] > 0][:3]
    losers = [ticker for ticker in reversed(tickers) if ticker["percent"] < 0][:3]
    for ticker in gainers:
        movers.append(
            (
                f"+{ticker['percent']:.2f}%",
                f"{ticker['symbol']} {ticker['name']}",
                "",
            )
        )
    for ticker in losers:
        movers.append(
            (
                f"{ticker['percent']:.2f}%",
                f"{ticker['symbol']} {ticker['name']}",
                "",
            )
        )
    return movers


def market_news_needs_refresh() -> bool:
    return indices_needs_refresh()


def get_events() -> list[tuple[str, str, str]]:
    return []


def get_checklist_left() -> list[str]:
    return ["Trend is your friend", "Check support & resistance", "Confirm with volume"]


def get_checklist_right() -> list[str]:
    return ["Manage risk (set stop loss)", "Book partial profits", "Let winners run"]
