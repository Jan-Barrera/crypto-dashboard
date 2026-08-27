#!/usr/bin/env python3
"""Ichimoku crypto swing-trade screener.

1. Loads pairs from crypto_list.txt
2. Downloads latest Binance daily OHLCV
3. Screens for swing setups
4. Creates crypto_swingtrade (if needed) and upserts rows into Supabase
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv
from sqlalchemy.dialects.postgresql import insert as pg_insert

from config import (
    CACHE_DIR,
    asset_name,
    load_crypto_pairs,
    to_binance_symbol,
    to_display_pair,
)
from db.hist_data import fetch_klines

logger = logging.getLogger(__name__)

TENKAN = 9
KIJUN = 26
SENKOU = 52
EMA_SPAN = 21
RSI_PERIOD = 14
MACD_FAST = 12
MACD_SLOW = 26
MACD_SIGNAL = 9
BB_PERIOD = 20
BB_STD = 2
GOLDEN_FAST = 50
GOLDEN_SLOW = 200
NEAR_HIGH_LOOKBACK = 20
GAIN_LOOKBACK = 5
GAIN_5D_MAX = 0.12
CLOSE_RANGE_TOP = 0.75
RESISTANCE_LOOKBACK_DAYS = 365
PIVOT_WINDOW = 5
ADTV_PERIOD = 20
ADX_PERIOD = 14
ADX_MIN = 25
RSI_MAX = 90
MIN_AVG_QUOTE_VOLUME = 1_000_000
STOP_ATR_MULTIPLE = 2.0
TARGET_ATR_MULTIPLE = 3.0
ATR_PERIOD = 14

CONFIRM_COLUMNS = [
    "above_ema21",
    "rsi_bullish",
    "macd_bullish",
    "golden_cross",
    "bollinger_bullish",
    "adx_bullish",
    "liquid_pair",
]
BULLISH_COLUMNS = [
    "above_cloud",
    "bullish_cloud",
    "tenkan_above_kijun",
    "tk_cross_bullish",
    "chikou_bullish",
]

ROOT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT_DIR / "swingtrades"
TABLE_NAME = "crypto_swingtrade"

CREATE_TABLE_SQL = f"""
CREATE TABLE IF NOT EXISTS {TABLE_NAME} (
    id BIGSERIAL PRIMARY KEY,
    pair TEXT NOT NULL,
    name TEXT,
    date DATE NOT NULL,
    signal_date DATE NOT NULL,
    close DOUBLE PRECISION,
    support DOUBLE PRECISION,
    support_dist_pct DOUBLE PRECISION,
    resistance DOUBLE PRECISION,
    resistance_dist_pct DOUBLE PRECISION,
    stop_loss DOUBLE PRECISION,
    take_profit DOUBLE PRECISION,
    atr DOUBLE PRECISION,
    ema21 DOUBLE PRECISION,
    rsi DOUBLE PRECISION,
    macd DOUBLE PRECISION,
    macd_signal DOUBLE PRECISION,
    adx DOUBLE PRECISION,
    plus_di DOUBLE PRECISION,
    minus_di DOUBLE PRECISION,
    adtv DOUBLE PRECISION,
    volume DOUBLE PRECISION,
    confirm_count INTEGER,
    adx_bullish BOOLEAN,
    obv_bullish BOOLEAN,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    UNIQUE (pair, signal_date)
);
"""

DB_COLUMNS = [
    "pair",
    "name",
    "date",
    "signal_date",
    "close",
    "support",
    "support_dist_pct",
    "resistance",
    "resistance_dist_pct",
    "stop_loss",
    "take_profit",
    "atr",
    "ema21",
    "rsi",
    "macd",
    "macd_signal",
    "adx",
    "plus_di",
    "minus_di",
    "adtv",
    "volume",
    "confirm_count",
    "adx_bullish",
    "obv_bullish",
]


def output_path_for_date(signal_date) -> Path:
    date_label = pd.Timestamp(signal_date).strftime("%Y-%m-%d")
    return OUTPUT_DIR / f"swingtrade_signals_{date_label}.csv"


def _drop_unusable_daily_candle(prices: pd.DataFrame) -> pd.DataFrame:
    """Drop the current UTC daily candle only while it is still too fresh.

    Binance 1d bars are keyed by open time and close at 00:00 UTC.
    Always dropping ``open_time >= today`` made midday/evening runs look a full
    day stale (e.g. 22:00 UTC still screened on yesterday).

    Keep the forming candle once it has enough age; otherwise use the prior
    completed bar (important for the 01:00 UTC GitHub Action).
    """
    if prices.empty:
        return prices

    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    last_open = pd.Timestamp(str(prices.index[-1])).tz_localize(None)
    candle_close = last_open + pd.Timedelta(days=1)
    is_complete = now >= candle_close
    if is_complete:
        return prices

    # Forming candle: keep it after 12h so later-day runs use latest prices.
    min_age = pd.Timedelta(hours=12)
    if now - last_open >= min_age:
        return prices

    return prices.iloc[:-1].copy()


def download_latest_prices(pairs: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Force-download latest Binance daily klines for each pair in crypto_list."""
    pairs = pairs or load_crypto_pairs()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    prices_by_pair: dict[str, pd.DataFrame] = {}
    now = pd.Timestamp.now(tz="UTC")

    for pair in pairs:
        display_pair = to_display_pair(pair)
        binance_symbol = to_binance_symbol(pair)
        try:
            print(f"Downloading {display_pair} from Binance...")
            raw = fetch_klines(binance_symbol)
            if raw.empty:
                print(f"  Skipping {display_pair}: no candles returned")
                continue

            raw_last = pd.Timestamp(str(raw.index.max()))
            prices = _drop_unusable_daily_candle(raw)
            if prices.empty:
                print(f"  Skipping {display_pair}: no usable completed candles")
                continue

            used_last = pd.Timestamp(str(prices.index.max()))
            cache_path = CACHE_DIR / f"{binance_symbol}_data.csv"
            # Cache the full Binance response (includes forming candle).
            raw.to_csv(cache_path)
            prices_by_pair[display_pair] = prices

            forming = raw_last > used_last
            note = (
                f"Binance last open {raw_last.date()} still forming "
                f"(UTC now {now.strftime('%Y-%m-%d %H:%M')}); "
                f"screening uses {used_last.date()}"
                if forming
                else f"screening through {used_last.date()}"
            )
            print(f"  {display_pair}: {len(prices)} rows, {note}")
        except Exception as exc:
            print(f"  Failed {display_pair}: {exc}")
            logger.warning("Failed to download %s: %s", display_pair, exc)

    print(f"Price load: {len(prices_by_pair)}/{len(pairs)} pairs ready")
    return prices_by_pair


def compute_ichimoku(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    high = result["High"]
    low = result["Low"]
    close = result["Close"]

    result["tenkan_sen"] = (high.rolling(TENKAN).max() + low.rolling(TENKAN).min()) / 2
    result["kijun_sen"] = (high.rolling(KIJUN).max() + low.rolling(KIJUN).min()) / 2
    result["senkou_span_a"] = ((result["tenkan_sen"] + result["kijun_sen"]) / 2).shift(KIJUN)
    result["senkou_span_b"] = (
        (high.rolling(SENKOU).max() + low.rolling(SENKOU).min()) / 2
    ).shift(KIJUN)
    result["chikou_span"] = close.shift(-KIJUN)
    return result


def ichimoku_signals(df: pd.DataFrame) -> pd.Series:
    latest = df.iloc[-1]
    prev = df.iloc[-2]
    cloud_top = max(latest["senkou_span_a"], latest["senkou_span_b"])
    cloud_bottom = min(latest["senkou_span_a"], latest["senkou_span_b"])

    return pd.Series(
        {
            "date": df.index[-1],
            "close": latest["Close"],
            "tenkan_sen": latest["tenkan_sen"],
            "kijun_sen": latest["kijun_sen"],
            "above_cloud": latest["Close"] > cloud_top,
            "below_cloud": latest["Close"] < cloud_bottom,
            "bullish_cloud": latest["senkou_span_a"] > latest["senkou_span_b"],
            "tenkan_above_kijun": latest["tenkan_sen"] > latest["kijun_sen"],
            "tk_cross_bullish": (
                latest["tenkan_sen"] > latest["kijun_sen"]
                and prev["tenkan_sen"] <= prev["kijun_sen"]
            ),
            "tk_cross_bearish": (
                latest["tenkan_sen"] < latest["kijun_sen"]
                and prev["tenkan_sen"] >= prev["kijun_sen"]
            ),
            "chikou_bullish": latest["Close"] > df["Close"].iloc[-KIJUN - 1],
        }
    )


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    close = result["Close"]
    result["ema21"] = close.ewm(span=EMA_SPAN, adjust=False).mean()

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / RSI_PERIOD, adjust=False).mean()
    result["rsi"] = 100 - (100 / (1 + avg_gain / avg_loss))

    ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
    ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
    result["macd"] = ema_fast - ema_slow
    result["macd_signal"] = result["macd"].ewm(span=MACD_SIGNAL, adjust=False).mean()
    result["macd_hist"] = result["macd"] - result["macd_signal"]
    result["sma50"] = close.rolling(GOLDEN_FAST).mean()
    result["sma200"] = close.rolling(GOLDEN_SLOW).mean()

    bb_middle = close.rolling(BB_PERIOD).mean()
    bb_std = close.rolling(BB_PERIOD).std()
    result["bb_middle"] = bb_middle
    result["bb_upper"] = bb_middle + BB_STD * bb_std
    result["bb_lower"] = bb_middle - BB_STD * bb_std

    previous_close = close.shift(1)
    true_range = pd.concat(
        [
            result["High"] - result["Low"],
            (result["High"] - previous_close).abs(),
            (result["Low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    result["atr"] = true_range.ewm(alpha=1 / ATR_PERIOD, adjust=False).mean()

    volume = pd.Series(
        pd.to_numeric(result["Volume"], errors="coerce"),
        index=result.index,
        dtype="float64",
    ).fillna(0.0)
    result["Volume"] = volume
    result["quote_volume"] = close * result["Volume"]
    result["adtv_volume"] = result["Volume"].rolling(ADTV_PERIOD).mean()
    result["adtv"] = result["quote_volume"].rolling(ADTV_PERIOD).mean()
    result["rel_volume"] = result["Volume"] / result["adtv_volume"].replace(0, float("nan"))

    high = result["High"]
    low = result["Low"]
    hl_range = (high - low).replace(0, float("nan"))
    clv_raw = ((close - low) - (high - close)) / hl_range
    clv = pd.Series(
        pd.to_numeric(clv_raw, errors="coerce"),
        index=result.index,
        dtype="float64",
    ).fillna(0.0)
    result["ad"] = (clv * result["Volume"]).cumsum()
    result["ad_sma20"] = result["ad"].rolling(ADTV_PERIOD).mean()

    direction = close.diff().fillna(0.0)
    signed_volume = result["Volume"].where(direction > 0, 0.0) - result["Volume"].where(
        direction < 0, 0.0
    )
    result["obv"] = signed_volume.cumsum()
    result["obv_sma20"] = result["obv"].rolling(ADTV_PERIOD).mean()
    return result


def find_nearest_resistance(df: pd.DataFrame) -> float:
    latest_close = float(df["Close"].iloc[-1])
    candidates: list[float] = []

    if "bb_upper" in df.columns:
        bb_upper = float(df["bb_upper"].iloc[-1])
        if pd.notna(bb_upper) and bb_upper > latest_close:
            candidates.append(bb_upper)

    high = pd.Series(df["High"], index=df.index, dtype="float64")
    span = PIVOT_WINDOW * 2 + 1
    is_pivot = (high == high.rolling(span, center=True).max()).fillna(False)
    is_pivot.iloc[-PIVOT_WINDOW:] = False
    swing_highs = high.loc[is_pivot.astype(bool)]
    above_swings = swing_highs.loc[swing_highs > latest_close]
    if len(above_swings) > 0:
        candidates.append(float(above_swings.min()))

    cutoff = pd.Timestamp(str(df.index[-1])) - pd.Timedelta(days=RESISTANCE_LOOKBACK_DAYS)
    window = df.loc[df.index >= cutoff]
    period_high = float(window["High"].max())
    if period_high > latest_close:
        candidates.append(period_high)

    return min(candidates) if candidates else float("nan")


def find_nearest_support(df: pd.DataFrame) -> float:
    latest_close = float(df["Close"].iloc[-1])
    candidates: list[float] = []

    if "bb_lower" in df.columns:
        bb_lower = float(df["bb_lower"].iloc[-1])
        if pd.notna(bb_lower) and bb_lower < latest_close:
            candidates.append(bb_lower)

    low = pd.Series(df["Low"], index=df.index, dtype="float64")
    span = PIVOT_WINDOW * 2 + 1
    is_pivot = (low == low.rolling(span, center=True).min()).fillna(False)
    is_pivot.iloc[-PIVOT_WINDOW:] = False
    swing_lows = low.loc[is_pivot.astype(bool)]
    below_swings = swing_lows.loc[swing_lows < latest_close]
    if len(below_swings) > 0:
        candidates.append(float(below_swings.max()))

    cutoff = pd.Timestamp(str(df.index[-1])) - pd.Timedelta(days=RESISTANCE_LOOKBACK_DAYS)
    window = df.loc[df.index >= cutoff]
    period_low = float(window["Low"].min())
    if period_low < latest_close:
        candidates.append(period_low)

    return max(candidates) if candidates else float("nan")


def confirmation_signals(df: pd.DataFrame) -> pd.Series:
    latest = df.iloc[-1]
    close = df["Close"]

    day_range = latest["High"] - latest["Low"]
    range_position = (latest["Close"] - latest["Low"]) / day_range if day_range > 0 else 1.0
    gain_5d = close.pct_change(GAIN_LOOKBACK).iloc[-1]
    resistance = find_nearest_resistance(df)
    resistance_dist_pct = (
        (resistance / latest["Close"] - 1) * 100 if pd.notna(resistance) else float("nan")
    )
    support = find_nearest_support(df)
    support_dist_pct = (
        (support / latest["Close"] - 1) * 100 if pd.notna(support) else float("nan")
    )

    ad = float(latest["ad"]) if pd.notna(latest.get("ad")) else float("nan")
    ad_sma20 = float(latest["ad_sma20"]) if pd.notna(latest.get("ad_sma20")) else float("nan")
    prev_ad = float(df["ad"].iloc[-2]) if len(df) > 1 and pd.notna(df["ad"].iloc[-2]) else float("nan")
    ad_rising = pd.notna(ad) and pd.notna(prev_ad) and ad > prev_ad
    ad_bullish = pd.notna(ad) and pd.notna(ad_sma20) and ad > ad_sma20 and ad_rising

    obv = float(latest["obv"]) if pd.notna(latest.get("obv")) else float("nan")
    obv_sma20 = float(latest["obv_sma20"]) if pd.notna(latest.get("obv_sma20")) else float("nan")
    prev_obv = (
        float(df["obv"].iloc[-2]) if len(df) > 1 and pd.notna(df["obv"].iloc[-2]) else float("nan")
    )
    obv_rising = pd.notna(obv) and pd.notna(prev_obv) and obv > prev_obv
    obv_bullish = pd.notna(obv) and pd.notna(obv_sma20) and obv > obv_sma20 and obv_rising

    atr = float(latest["atr"]) if pd.notna(latest.get("atr")) else float("nan")
    stop_loss = latest["Close"] - STOP_ATR_MULTIPLE * atr if pd.notna(atr) else float("nan")
    take_profit = latest["Close"] + TARGET_ATR_MULTIPLE * atr if pd.notna(atr) else float("nan")

    return pd.Series(
        {
            "date": df.index[-1],
            "close": latest["Close"],
            "support": support,
            "support_dist_pct": support_dist_pct,
            "resistance": resistance,
            "resistance_dist_pct": resistance_dist_pct,
            "ema21": latest["ema21"],
            "rsi": latest["rsi"],
            "macd": latest["macd"],
            "macd_signal": latest["macd_signal"],
            "atr": atr,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "gain_5d": gain_5d,
            "range_position": range_position,
            "volume": float(latest["Volume"]),
            "adtv": float(latest["adtv"]) if pd.notna(latest.get("adtv")) else float("nan"),
            "rel_volume": (
                float(latest["rel_volume"]) if pd.notna(latest.get("rel_volume")) else float("nan")
            ),
            "ad": ad,
            "ad_bullish": ad_bullish,
            "obv": obv,
            "obv_sma20": obv_sma20,
            "obv_bullish": obv_bullish,
            "above_ema21": latest["Close"] > latest["ema21"],
            "rsi_bullish": latest["rsi"] > 50,
            "macd_bullish": latest["macd"] > latest["macd_signal"],
            "golden_cross": latest["sma50"] > latest["sma200"],
            "bollinger_bullish": latest["Close"] > latest["bb_middle"],
            "liquid_pair": (
                pd.notna(latest.get("adtv")) and latest["adtv"] >= MIN_AVG_QUOTE_VOLUME
            ),
            "highest_close_20": latest["Close"] >= close.iloc[-NEAR_HIGH_LOOKBACK:].max(),
            "close_top_25_range": range_position >= CLOSE_RANGE_TOP,
            "gain_5d_below_12": gain_5d < GAIN_5D_MAX,
        }
    )


def compute_adx(df: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
    result = df.copy()
    high = result["High"]
    low = result["Low"]
    close = result["Close"]

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)

    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_dm_smooth = plus_dm.ewm(alpha=1 / period, adjust=False).mean()
    minus_dm_smooth = minus_dm.ewm(alpha=1 / period, adjust=False).mean()

    result["plus_di"] = 100 * plus_dm_smooth / atr
    result["minus_di"] = 100 * minus_dm_smooth / atr
    di_sum = (result["plus_di"] + result["minus_di"]).replace(0, float("nan"))
    dx = 100 * (result["plus_di"] - result["minus_di"]).abs() / di_sum
    result["adx"] = dx.ewm(alpha=1 / period, adjust=False).mean()
    return result


def build_ichimoku_signals(all_prices: dict[str, pd.DataFrame]) -> pd.DataFrame:
    signals = []
    for pair, prices in sorted(all_prices.items()):
        if len(prices) < SENKOU + KIJUN + 1:
            continue

        ichimoku_df = compute_ichimoku(prices)
        required = ["tenkan_sen", "kijun_sen", "senkou_span_a", "senkou_span_b"]
        if ichimoku_df[required].iloc[-1].isna().any():
            continue

        row = ichimoku_signals(ichimoku_df)
        row.name = pair
        signals.append(row)

    signals_df = pd.DataFrame(signals)
    if signals_df.empty:
        return signals_df
    signals_df.index.name = "pair"
    signals_df["bullish_count"] = signals_df[BULLISH_COLUMNS].sum(axis=1)
    return signals_df


def build_confirmations(
    all_prices: dict[str, pd.DataFrame], bullish_pairs: pd.DataFrame
) -> pd.DataFrame:
    confirmations = []
    minimum_history = max(GOLDEN_SLOW, MACD_SLOW + MACD_SIGNAL, ADTV_PERIOD)
    for pair in bullish_pairs.index:
        prices = all_prices[pair]
        if len(prices) < minimum_history:
            continue

        indicators = compute_indicators(prices)
        required = ["ema21", "rsi", "macd", "bb_middle", "atr", "adtv"]
        if indicators[required].iloc[-1].isna().any():
            continue

        row = confirmation_signals(indicators)
        row.name = pair
        confirmations.append(row)

    confirmations_df = pd.DataFrame(confirmations)
    if confirmations_df.empty:
        return confirmations_df
    confirmations_df.index.name = "pair"
    return confirmations_df


def add_adx_confirmation(
    all_prices: dict[str, pd.DataFrame], confirmations: pd.DataFrame
) -> pd.DataFrame:
    if confirmations.empty:
        return confirmations

    adx_values = []
    plus_di_values = []
    minus_di_values = []
    adx_bullish_values = []

    for pair in confirmations.index:
        prices = all_prices[pair]
        latest = compute_adx(prices).iloc[-1]
        adx_ok = pd.notna(latest["adx"]) and latest["adx"] >= ADX_MIN
        di_bullish = pd.notna(latest["plus_di"]) and latest["plus_di"] > latest["minus_di"]
        adx_values.append(latest["adx"])
        plus_di_values.append(latest["plus_di"])
        minus_di_values.append(latest["minus_di"])
        adx_bullish_values.append(bool(adx_ok and di_bullish))

    out = confirmations.copy()
    out["adx"] = adx_values
    out["plus_di"] = plus_di_values
    out["minus_di"] = minus_di_values
    out["adx_bullish"] = adx_bullish_values
    out["confirm_count"] = out[CONFIRM_COLUMNS].sum(axis=1)
    return out


def apply_filters(confirmations: pd.DataFrame) -> pd.DataFrame:
    if confirmations.empty:
        return confirmations

    out = confirmations.copy()
    out = out.loc[out["rsi"] <= RSI_MAX]
    out = out.loc[out["liquid_pair"] == True]  # noqa: E712
    out = out.loc[out["confirm_count"] > 3]
    out = out.loc[out["adx_bullish"] == True]  # noqa: E712
    out = out.loc[out["obv_bullish"] == True]  # noqa: E712
    assert isinstance(out, pd.DataFrame)
    return out


def prepare_swingtrade_rows(filtered: pd.DataFrame) -> pd.DataFrame:
    if filtered.empty:
        return pd.DataFrame(columns=DB_COLUMNS)

    out = filtered.sort_values(
        ["confirm_count", "adtv"], ascending=[False, False]
    ).reset_index()
    if "pair" not in out.columns:
        raise ValueError("Expected pair index/column on filtered swingtrade frame")

    out["pair"] = out["pair"].astype(str)
    out["name"] = out["pair"].map(asset_name)
    out["date"] = pd.to_datetime(out["date"]).dt.date
    out["signal_date"] = out["date"]
    confirm_count = pd.Series(
        pd.to_numeric(out["confirm_count"], errors="coerce"),
        index=out.index,
        dtype="float64",
    ).fillna(0)
    out["confirm_count"] = confirm_count.astype(int)
    out["adx_bullish"] = out["adx_bullish"].astype(bool)
    out["obv_bullish"] = out["obv_bullish"].astype(bool)
    return out.loc[:, DB_COLUMNS].copy()


def save_swingtrade_csv(filtered: pd.DataFrame) -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if filtered.empty:
        signal_date = pd.Timestamp.now(tz="UTC").normalize()
        output_path = output_path_for_date(signal_date)
        print(f"No rows to save to {output_path}")
        return 0

    out = filtered.sort_values(
        ["confirm_count", "adtv"], ascending=[False, False]
    ).reset_index()
    signal_date = pd.to_datetime(out["date"]).max()
    out["signal_date"] = pd.to_datetime(out["date"]).dt.strftime("%Y-%m-%d")
    output_path = output_path_for_date(signal_date)
    out.to_csv(output_path, index=False)
    print(f"Saved {len(out)} rows to {output_path.resolve()}")
    return len(out)


def get_engine() -> sa.Engine:
    load_dotenv(ROOT_DIR / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise ValueError("DATABASE_URL not found in .env")
    return sa.create_engine(database_url)


def ensure_crypto_swingtrade_table(engine: sa.Engine) -> None:
    with engine.begin() as conn:
        conn.execute(sa.text(CREATE_TABLE_SQL))
    print(f"Ensured table exists: {TABLE_NAME}")


def upsert_crypto_swingtrade(engine: sa.Engine, rows: pd.DataFrame) -> int:
    if rows.empty:
        print("No swingtrade rows to insert.")
        return 0

    payload = rows.where(pd.notnull(rows), None).to_dict(orient="records")
    table = sa.table(
        TABLE_NAME,
        *[sa.column(name) for name in DB_COLUMNS],
    )
    stmt = pg_insert(table).values(payload)
    update_cols = {
        column.name: stmt.excluded[column.name]
        for column in table.c
        if column.name not in {"pair", "signal_date"}
    }
    stmt = stmt.on_conflict_do_update(
        index_elements=["pair", "signal_date"],
        set_=update_cols,
    )
    with engine.begin() as conn:
        conn.execute(stmt)
    print(f"Upserted {len(payload)} rows into {TABLE_NAME}")
    return len(payload)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    pairs = load_crypto_pairs()
    print(f"Loaded {len(pairs)} pairs from crypto_list.txt")

    all_prices = download_latest_prices(pairs)
    if not all_prices:
        print("No price data downloaded.")
        return

    signals = build_ichimoku_signals(all_prices)
    if signals.empty:
        print("No Ichimoku signals computed.")
        return

    bullish_pairs = signals.loc[signals["bullish_count"] > 2]
    if not isinstance(bullish_pairs, pd.DataFrame):
        print("No bullish pairs.")
        return
    print(f"Ichimoku bullish pairs: {len(bullish_pairs)}")

    confirmations = build_confirmations(all_prices, bullish_pairs)
    confirmations = add_adx_confirmation(all_prices, confirmations)
    filtered = apply_filters(confirmations)

    print("\nFiltered crypto swing candidates:")
    if filtered.empty:
        print("(none)")
    else:
        display = filtered.sort_values(
            ["confirm_count", "adtv"], ascending=[False, False]
        )[
            [
                "date",
                "close",
                "support",
                "resistance",
                "stop_loss",
                "take_profit",
                "adx",
                "adtv",
                "rsi",
                "macd",
                "confirm_count",
            ]
        ]
        print(display.round(2).to_string())

    save_swingtrade_csv(filtered)
    rows = prepare_swingtrade_rows(filtered)

    engine = get_engine()
    ensure_crypto_swingtrade_table(engine)
    upsert_crypto_swingtrade(engine, rows)


if __name__ == "__main__":
    main()
