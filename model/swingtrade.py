from __future__ import annotations

import logging
import os
from functools import lru_cache
from pathlib import Path

import pandas as pd
import sqlalchemy as sa
from dotenv import load_dotenv

from config import ROOT_DIR, asset_name, format_usd

logger = logging.getLogger(__name__)

TABLE_NAME = "crypto_swingtrade"

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


@lru_cache(maxsize=1)
def get_engine() -> sa.Engine:
    load_dotenv(Path(ROOT_DIR) / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        try:
            import streamlit as st

            database_url = st.secrets.get("DATABASE_URL")
        except Exception:
            database_url = None
    if not database_url:
        raise ValueError("DATABASE_URL not found in .env or Streamlit secrets")
    return sa.create_engine(database_url)


def fetch_latest_swingtrade(db_engine: sa.Engine | None = None) -> pd.DataFrame:
    """Load the newest signal_date rows from crypto_swingtrade."""
    engine = db_engine or get_engine()
    query = sa.text(
        f"""
        select
            pair,
            name,
            date,
            signal_date,
            close,
            support,
            resistance,
            stop_loss,
            take_profit,
            rsi,
            adx,
            confirm_count,
            adtv
        from {TABLE_NAME}
        where signal_date = (select max(signal_date) from {TABLE_NAME})
        order by confirm_count desc nulls last, adtv desc nulls last, pair
        """
    )
    frame = pd.read_sql(query, engine, parse_dates=["date", "signal_date"])
    if frame.empty:
        logger.warning("Supabase table %s returned no rows for latest signal_date", TABLE_NAME)
        return frame

    frame = frame.rename(columns={"pair": "symbol"})
    logger.info(
        "Loaded %s swingtrade rows from %s through %s",
        len(frame),
        TABLE_NAME,
        pd.to_datetime(frame["signal_date"]).max().date(),
    )
    return frame


def get_swingtrade_dataframe() -> pd.DataFrame:
    """Return the latest swingtrade rows from Supabase."""
    return fetch_latest_swingtrade()


def _cell(row: pd.Series, key: str) -> object:
    value = row.loc[key] if key in row.index else None
    if isinstance(value, pd.Series):
        return None if value.empty else value.iloc[0]
    return value


def _is_missing(value: object) -> bool:
    if value is None:
        return True
    result = pd.isna(value)
    return bool(result) if isinstance(result, bool) else False


def _as_float(value: object) -> float | None:
    if _is_missing(value):
        return None
    return float(f"{value}")


def get_swing_trade_watchlist() -> pd.DataFrame:
    """Return swing trade watchlist formatted for the UI table."""
    frame = get_swingtrade_dataframe()
    if frame.empty:
        return pd.DataFrame(columns=WATCHLIST_COLUMNS)

    rows = []
    for _, row in frame.iterrows():
        trade_date = _cell(row, "date")
        date_label = (
            pd.Timestamp(str(trade_date)).strftime("%Y-%m-%d")
            if not _is_missing(trade_date)
            else ""
        )
        rsi = _as_float(_cell(row, "rsi"))
        adx = _as_float(_cell(row, "adx"))
        symbol = str(_cell(row, "symbol") or "")
        name_raw = _cell(row, "name")
        rows.append(
            {
                "Symbol": symbol,
                "Name": str(name_raw) if name_raw else asset_name(symbol),
                "Date": date_label,
                "Close": format_usd(_cell(row, "close")),
                "Support": format_usd(_cell(row, "support")),
                "Resistance": format_usd(_cell(row, "resistance")),
                "Stop Loss": format_usd(_cell(row, "stop_loss")),
                "Take Profit": format_usd(_cell(row, "take_profit")),
                "RSI": f"{rsi:.1f}" if rsi is not None else "",
                "ADX": f"{adx:.1f}" if adx is not None else "",
            }
        )
    return pd.DataFrame(rows, columns=WATCHLIST_COLUMNS)
