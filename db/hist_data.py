from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from config import (
    CACHE_DIR,
    CACHE_TTL_SECONDS,
    KLINE_INTERVAL,
    KLINE_LIMIT,
    LOOKBACK_DAYS,
    to_binance_symbol,
    to_display_pair,
)
from db.binance import binance_get

logger = logging.getLogger(__name__)

_REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
_KLINE_COLUMNS = [
    "open_time",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "close_time",
    "QuoteVolume",
    "Trades",
    "TakerBuyBase",
    "TakerBuyQuote",
    "Ignore",
]


def clip_to_lookback(prices: pd.DataFrame, days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    if prices.empty:
        return prices.copy()
    last_date = pd.Timestamp(str(prices.index.max()))
    cutoff = last_date - pd.Timedelta(days=days)
    clipped = prices.loc[prices.index >= cutoff].copy()
    assert isinstance(clipped, pd.DataFrame)
    return clipped


def load_cached(path: str) -> pd.DataFrame | None:
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    age_seconds = time.time() - os.path.getmtime(path)
    if age_seconds >= CACHE_TTL_SECONDS:
        return None
    try:
        cached = pd.read_csv(path, parse_dates=["date"], index_col="date")
    except (pd.errors.EmptyDataError, ValueError, KeyError):
        return None
    if not isinstance(cached, pd.DataFrame):
        return None
    if cached.empty or not _REQUIRED_COLUMNS.issubset(cached.columns):
        return None
    cached = cached[~cached.index.duplicated(keep="last")].sort_index()
    assert isinstance(cached, pd.DataFrame)
    logger.info(
        "Using cached Binance klines (%s rows, age %.0fs): %s",
        len(cached),
        age_seconds,
        path,
    )
    return clip_to_lookback(cached)


def fetch_klines(
    symbol: str,
    interval: str = KLINE_INTERVAL,
    limit: int = KLINE_LIMIT,
) -> pd.DataFrame:
    binance_symbol = to_binance_symbol(symbol)
    payload = binance_get(
        "/api/v3/klines",
        {"symbol": binance_symbol, "interval": interval, "limit": int(limit)},
    )
    if not payload:
        raise ValueError(f"Binance returned no klines for {binance_symbol}")

    prices = pd.DataFrame(payload, columns=_KLINE_COLUMNS)
    prices["date"] = pd.to_datetime(prices["open_time"], unit="ms")
    prices = prices.set_index("date")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        prices[column] = pd.to_numeric(prices[column], errors="coerce")
    prices = prices.dropna(subset=["Open", "High", "Low", "Close"])
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    assert isinstance(prices, pd.DataFrame)
    if prices.empty:
        raise ValueError(f"Binance klines for {binance_symbol} had no usable OHLC rows")
    ohlcv = prices.loc[:, ["Open", "High", "Low", "Close", "Volume"]]
    assert isinstance(ohlcv, pd.DataFrame)
    return clip_to_lookback(ohlcv)


def get_hist_data(symbol: str) -> pd.DataFrame:
    binance_symbol = to_binance_symbol(symbol)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    csv_filename = str(CACHE_DIR / f"{binance_symbol}_data.csv")
    cached = load_cached(csv_filename)
    if cached is not None:
        return cached

    prices = fetch_klines(binance_symbol)
    prices.to_csv(csv_filename)
    last_date = pd.Timestamp(str(prices.index.max()))
    logger.info(
        "Fetched and cached Binance klines: %s (%s rows through %s)",
        csv_filename,
        len(prices),
        last_date.date(),
    )
    return prices


def get_hist_data_many(symbols: list[str], max_workers: int = 8) -> dict[str, pd.DataFrame]:
    results: dict[str, pd.DataFrame] = {}
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(get_hist_data, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                results[to_display_pair(symbol)] = future.result()
            except Exception as exc:
                logger.warning("Failed to load klines for %s: %s", symbol, exc)
    return results
