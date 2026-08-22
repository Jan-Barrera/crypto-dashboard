from __future__ import annotations

import datetime
import logging
import os

import pandas as pd

from config import (
    CACHE_DIR,
    CACHE_TTL_SECONDS,
    asset_name,
    format_usd,
    load_crypto_pairs,
    to_display_pair,
)
from db.hist_data import get_hist_data_many

logger = logging.getLogger(__name__)

CACHE_PATH = CACHE_DIR / "swingtrade_latest.csv"

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

WATCHLIST_COLUMNS = [
    "Symbol",
    "Name",
    "Date",
    "Close",
    "Support",
    "Resistance",
    "Stop Loss",
    "Take Profit",
    "RSI",
    "ADX",
]

RAW_COLUMNS = [
    "symbol",
    "name",
    "date",
    "close",
    "support",
    "resistance",
    "stop_loss",
    "take_profit",
    "rsi",
    "adx",
    "confirm_count",
    "adtv",
]


def load_cached_swingtrade(path: os.PathLike[str] | str = CACHE_PATH) -> pd.DataFrame | None:
    path = os.fspath(path)
    if not os.path.exists(path) or os.path.getsize(path) == 0:
        return None
    age_seconds = datetime.datetime.now().timestamp() - os.path.getmtime(path)
    if age_seconds >= CACHE_TTL_SECONDS:
        return None
    try:
        cached = pd.read_csv(path, parse_dates=["date"])
    except (pd.errors.EmptyDataError, ValueError, KeyError):
        return None
    required = {
        "symbol",
        "date",
        "close",
        "support",
        "resistance",
        "stop_loss",
        "take_profit",
        "rsi",
        "adx",
        "confirm_count",
    }
    if cached.empty or not required.issubset(cached.columns):
        return None
    logger.info("Using cached swingtrade (%s rows): %s", len(cached), path)
    return cached


def _drop_in_progress_candle(prices: pd.DataFrame) -> pd.DataFrame:
    """Discard today's incomplete daily candle, matching Jupyter screener behavior."""
    if prices.empty:
        return prices
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    if prices.index[-1] >= today:
        return prices.iloc[:-1].copy()
    return prices


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

    result["Volume"] = pd.to_numeric(result["Volume"], errors="coerce").fillna(0.0)
    result["quote_volume"] = close * result["Volume"]
    result["adtv_volume"] = result["Volume"].rolling(ADTV_PERIOD).mean()
    result["adtv"] = result["quote_volume"].rolling(ADTV_PERIOD).mean()
    result["rel_volume"] = result["Volume"] / result["adtv_volume"].replace(0, float("nan"))

    high = result["High"]
    low = result["Low"]
    hl_range = (high - low).replace(0, float("nan"))
    clv = ((close - low) - (high - close)) / hl_range
    clv = pd.to_numeric(clv, errors="coerce").fillna(0.0)
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
        bb_upper = df["bb_upper"].iloc[-1]
        if pd.notna(bb_upper) and bb_upper > latest_close:
            candidates.append(float(bb_upper))

    high = df["High"]
    span = PIVOT_WINDOW * 2 + 1
    is_pivot = high == high.rolling(span, center=True).max()
    is_pivot = is_pivot.fillna(False)
    is_pivot.iloc[-PIVOT_WINDOW:] = False
    swing_highs = high[is_pivot]
    above_swings = swing_highs[swing_highs > latest_close]
    if not above_swings.empty:
        candidates.append(float(above_swings.min()))

    cutoff = df.index[-1] - pd.Timedelta(days=RESISTANCE_LOOKBACK_DAYS)
    window = df.loc[df.index >= cutoff]
    period_high = float(window["High"].max())
    if period_high > latest_close:
        candidates.append(period_high)

    return min(candidates) if candidates else float("nan")


def find_nearest_support(df: pd.DataFrame) -> float:
    latest_close = float(df["Close"].iloc[-1])
    candidates: list[float] = []

    if "bb_lower" in df.columns:
        bb_lower = df["bb_lower"].iloc[-1]
        if pd.notna(bb_lower) and bb_lower < latest_close:
            candidates.append(float(bb_lower))

    low = df["Low"]
    span = PIVOT_WINDOW * 2 + 1
    is_pivot = low == low.rolling(span, center=True).min()
    is_pivot = is_pivot.fillna(False)
    is_pivot.iloc[-PIVOT_WINDOW:] = False
    swing_lows = low[is_pivot]
    below_swings = swing_lows[swing_lows < latest_close]
    if not below_swings.empty:
        candidates.append(float(below_swings.max()))

    cutoff = df.index[-1] - pd.Timedelta(days=RESISTANCE_LOOKBACK_DAYS)
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
    support = find_nearest_support(df)

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
            "resistance": resistance,
            "ema21": latest["ema21"],
            "rsi": latest["rsi"],
            "macd": latest["macd"],
            "macd_signal": latest["macd_signal"],
            "atr": atr,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "gain_5d": gain_5d,
            "range_position": range_position,
            "adtv": float(latest["adtv"]) if pd.notna(latest.get("adtv")) else float("nan"),
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
    out = out[out["rsi"] <= RSI_MAX]
    out = out[out["liquid_pair"] == True]  # noqa: E712
    out = out[out["confirm_count"] > 3]
    out = out[out["adx_bullish"] == True]  # noqa: E712
    out = out[out["obv_bullish"] == True]  # noqa: E712
    return out


def build_swingtrade_dataframe() -> pd.DataFrame:
    pairs = load_crypto_pairs()
    histories = get_hist_data_many(pairs)
    all_prices: dict[str, pd.DataFrame] = {}
    for pair in pairs:
        display_pair = to_display_pair(pair)
        prices = histories.get(display_pair)
        if prices is None or prices.empty:
            continue
        cleaned = _drop_in_progress_candle(prices)
        if cleaned.empty:
            continue
        all_prices[display_pair] = cleaned

    if not all_prices:
        raise ValueError("No Binance klines available to build the swing trade watchlist")

    signals = build_ichimoku_signals(all_prices)
    if signals.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    bullish_pairs = signals[signals["bullish_count"] > 2]
    confirmations = build_confirmations(all_prices, bullish_pairs)
    confirmations = add_adx_confirmation(all_prices, confirmations)
    filtered = apply_filters(confirmations)
    if filtered.empty:
        return pd.DataFrame(columns=RAW_COLUMNS)

    ranked = filtered.sort_values(
        ["confirm_count", "adtv"], ascending=[False, False]
    ).reset_index()
    rows: list[dict[str, object]] = []
    for _, row in ranked.iterrows():
        symbol = row["pair"]
        rows.append(
            {
                "symbol": symbol,
                "name": asset_name(symbol),
                "date": row["date"],
                "close": row["close"],
                "support": row["support"],
                "resistance": row["resistance"],
                "stop_loss": row["stop_loss"],
                "take_profit": row["take_profit"],
                "rsi": row["rsi"],
                "adx": row["adx"],
                "confirm_count": int(row["confirm_count"]),
                "adtv": row["adtv"],
            }
        )
    return pd.DataFrame(rows, columns=RAW_COLUMNS)


def get_swingtrade_dataframe() -> pd.DataFrame:
    cached = load_cached_swingtrade()
    if cached is not None:
        return cached

    frame = build_swingtrade_dataframe()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(CACHE_PATH, index=False)
    logger.info(
        "Fetched and cached swingtrade: %s (%s rows)",
        CACHE_PATH,
        len(frame),
    )
    return frame


def get_swing_trade_watchlist() -> pd.DataFrame:
    frame = get_swingtrade_dataframe()
    if frame.empty:
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)

    rows = []
    for _, row in frame.iterrows():
        trade_date = row["date"]
        date_label = (
            pd.Timestamp(trade_date).strftime("%Y-%m-%d")
            if pd.notna(trade_date)
            else ""
        )
        rsi = row["rsi"] if "rsi" in row and pd.notna(row["rsi"]) else None
        adx = row["adx"] if "adx" in row and pd.notna(row["adx"]) else None
        rows.append(
            {
                "Symbol": row["symbol"],
                "Name": row.get("name") or asset_name(row["symbol"]),
                "Date": date_label,
                "Close": format_usd(row["close"]),
                "Support": format_usd(row["support"]),
                "Resistance": format_usd(row["resistance"]),
                "Stop Loss": format_usd(row["stop_loss"]),
                "Take Profit": format_usd(row["take_profit"]),
                "RSI": f"{rsi:.1f}" if rsi is not None else "",
                "ADX": f"{adx:.1f}" if adx is not None else "",
            }
        )
    return pd.DataFrame(rows, columns=WATCHLIST_COLUMNS)
