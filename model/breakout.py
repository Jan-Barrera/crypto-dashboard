import logging
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from config import format_usd, to_display_pair
from db.hist_data import get_hist_data

logger = logging.getLogger(__name__)

pd.set_option("display.max_colwidth", None)
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "normal",
    "axes.labelweight": "bold",
    "figure.titleweight": "bold",
})
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Match ui/style.css dark palette (same as fibonacci plot)
_BG = "#0b0f19"
_PANEL = "#11172a"
_BORDER = "#1e2540"
_TEXT = "#e6e9ef"
_TEXT_MUTED = "#8b93a7"
_UP = "#26d07c"
_DOWN = "#f0555e"
_ACCENT = "#E5C100"
_MID = "#c7cce0"

_MARKET_COLORS = mpf.make_marketcolors(
    up=_UP,
    down=_DOWN,
    edge="inherit",
    wick="inherit",
    volume={"up": _UP, "down": _DOWN},
    inherit=True,
)

_STYLE = mpf.make_mpf_style(
    base_mpf_style="nightclouds",
    marketcolors=_MARKET_COLORS,
    facecolor=_PANEL,
    edgecolor=_BORDER,
    figcolor=_BG,
    gridcolor=_BORDER,
    gridstyle="--",
    gridaxis="both",
    rc={
        "font.family": "DejaVu Sans",
        "font.weight": "normal",
        "axes.labelweight": "bold",
        "figure.titleweight": "bold",
        "axes.labelcolor": _TEXT,
        "axes.edgecolor": _BORDER,
        "xtick.color": _TEXT_MUTED,
        "ytick.color": _TEXT_MUTED,
        "text.color": _TEXT,
        "figure.facecolor": _BG,
        "axes.facecolor": _PANEL,
    },
)


def _style_axes(axes) -> None:
    # Keep mplfinance's bar-index → datetime labels (datetime_format).
    # Do not apply mdates.DateFormatter here — x values are integers, not dates.
    for ax in axes:
        ax.set_facecolor(_PANEL)
        for spine in ax.spines.values():
            spine.set_color(_BORDER)
        ax.tick_params(colors=_TEXT_MUTED)
        ax.tick_params(axis="x", labelrotation=0)
        ax.yaxis.label.set_color(_TEXT)
        ax.xaxis.label.set_color(_TEXT)
        ax.title.set_color(_TEXT)
        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_ha("center")


@dataclass
class BreakoutChart:
    figure: plt.Figure
    summary: list[str]
    breakouts: pd.DataFrame
    pre_breakouts: pd.DataFrame


SETUP_LOOKBACK_DAYS = 90
SETUP_COLUMNS = ["Date", "Type", "Signal", "Close", "Level", "Note"]

_TYPE_PRIORITY = {
    "Bullish Breakout": 0,
    "Bearish Breakdown": 0,
    "Pre-Breakout (Bullish)": 0,
    "Pre-Breakdown (Bearish)": 0,
    "Breakout (confirmed)": 1,
    "Pre-breakout watch": 1,
}


def _format_price(value: float) -> str:
    return format_usd(value)


def _format_setup_row(
    date,
    type_: str,
    signal: str,
    close,
    level,
    note: str,
) -> dict[str, str]:
    return {
        "Date": pd.Timestamp(date).strftime("%Y-%m-%d") if pd.notna(date) else "",
        "Type": type_,
        "Signal": signal or "",
        "Close": _format_price(close),
        "Level": _format_price(level),
        "Note": note or "",
    }


def _dedupe_breakouts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ranked = df.copy()
    ranked["_priority"] = ranked["Type"].map(_TYPE_PRIORITY).fillna(2)
    ranked = ranked.sort_values(["Date", "_priority"], ascending=[False, True])
    return (
        ranked.drop_duplicates(subset=["Date", "Signal"], keep="first")
        .drop(columns="_priority")
        .reset_index(drop=True)
    )


def _dedupe_pre_breakouts(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    ranked = df.copy()
    ranked["_priority"] = ranked["Type"].map(_TYPE_PRIORITY).fillna(2)
    ranked = ranked.sort_values(["Date", "_priority"], ascending=[False, True])
    return ranked.drop_duplicates(subset=["Date"], keep="first").drop(columns="_priority").reset_index(drop=True)


def build_setup_tables(
    work: pd.DataFrame,
    breakouts_df: pd.DataFrame,
    pre_breakouts_df: pd.DataFrame,
    events_df: pd.DataFrame,
    lookback_days: int = SETUP_LOOKBACK_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = work.index[-1] - pd.Timedelta(days=lookback_days)
    breakout_rows: list[dict[str, str]] = []
    pre_rows: list[dict[str, str]] = []

    if not breakouts_df.empty:
        for _, row in breakouts_df[breakouts_df["date"] >= cutoff].iterrows():
            breakout_rows.append(
                _format_setup_row(
                    row["date"],
                    row["type"],
                    str(row.get("signal", "")),
                    row["close"],
                    row.get("level"),
                    row["note"],
                )
            )

    if not pre_breakouts_df.empty:
        for _, row in pre_breakouts_df[pre_breakouts_df["date"] >= cutoff].iterrows():
            pre_rows.append(
                _format_setup_row(
                    row["date"],
                    row["type"],
                    str(row.get("signal", "")),
                    row["close"],
                    row.get("level"),
                    row["note"],
                )
            )

    if not events_df.empty:
        for _, row in events_df[events_df["date"] >= cutoff].iterrows():
            setup_row = _format_setup_row(
                row["date"],
                row["type"],
                "Bullish" if row["type"] == "Breakout (confirmed)" else "Watch",
                row["close"],
                row.get("level"),
                row["note"],
            )
            if row["type"] == "Breakout (confirmed)":
                breakout_rows.append(setup_row)
            elif row["type"] == "Pre-breakout watch":
                pre_rows.append(setup_row)

    breakouts = (
        _dedupe_breakouts(pd.DataFrame(breakout_rows, columns=SETUP_COLUMNS))
        if breakout_rows
        else pd.DataFrame(columns=SETUP_COLUMNS)
    )
    pre_breakouts = (
        _dedupe_pre_breakouts(pd.DataFrame(pre_rows, columns=SETUP_COLUMNS))
        if pre_rows
        else pd.DataFrame(columns=SETUP_COLUMNS)
    )
    return breakouts, pre_breakouts


def create_breakout_plot(symbol: str, lookback_days: int = 365, chart_days: int = 120) -> BreakoutChart:
    """Load price data, calculate breakout signals, and return the chart."""
    pair = to_display_pair(symbol)
    if not pair:
        raise ValueError("Symbol is required")
    df = get_hist_data(pair)
    if lookback_days:
        cutoff = df.index.max() - pd.Timedelta(days=lookback_days)
        df = df[df.index >= cutoff].copy()
    # --- Indicator settings (TradingView-style) ---
    EMA_FAST, EMA_MID, EMA_SLOW = 20, 50, 200
    MACD_FAST, MACD_SLOW, MACD_SIGNAL = 12, 26, 9
    ADX_PERIOD = 14
    VOL_MA = 20
    # --- Consolidation / pre-breakout settings ---
    CONSOL_MIN_BARS = 10              # shortest consolidation window
    CONSOL_MAX_BARS = 25              # longest consolidation window
    BOX_RANGE_PCT_MAX = 0.15        # max adaptive box height vs midpoint
    NEAR_BOX_TOP_PCT = 0.025        # within 2.5% of box top = coiling
    VOL_QUIET_RATIO = 0.75          # volume below this × MA = "quiet"
    VOL_QUIET_MIN_BARS = 3          # consecutive quiet bars to flag zone
    MACD_DURATION_MIN = 5           # min bars of negative MACD hist for "duration" reset
    PRE_BREAKOUT_MIN_SCORE = 4      # of 6 checks below
    # Shared by rolling + box confirmed breakouts
    VOLUME_CONFIRM_RATIO = 1.5
    # --- Rolling breakout config (candlestick-v2) ---
    BREAKOUT_LOOKBACK = 20
    VOLUME_LOOKBACK = 20
    NEAR_LEVEL_PCT = 0.015
    COMPRESS_BARS = 5

    def detect_rolling_breakouts(price_df: pd.DataFrame):
        """Volume-confirmed rolling high/low breakouts + pre-breakout compression."""
        vol = price_df["Volume"].astype(float)
        o, h, l, c = price_df["Open"], price_df["High"], price_df["Low"], price_df["Close"]
        body = (c - o).abs()
        rng = h - l
        bullish = c > o
        bearish = c < o
        vol_avg = vol.rolling(VOLUME_LOOKBACK, min_periods=5).mean()
        vol_ratio = vol / vol_avg.replace(0, np.nan)
        roll_high = h.shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=10).max()
        roll_low = l.shift(1).rolling(BREAKOUT_LOOKBACK, min_periods=10).min()
        breakout_events = []
        pre_breakout_events = []
        def pick_resistance(close_px, rh):
            candidates = [(rh, f"{BREAKOUT_LOOKBACK}d high")]
            overhead = [(p, lbl) for p, lbl in candidates if p >= close_px * 0.995]
            if overhead:
                return min(overhead, key=lambda x: x[0] - close_px)
            return min(candidates, key=lambda x: x[0])
        def pick_support(close_px, rl):
            candidates = [(rl, f"{BREAKOUT_LOOKBACK}d low")]
            underfoot = [(p, lbl) for p, lbl in candidates if p <= close_px * 1.005]
            if underfoot:
                return max(underfoot, key=lambda x: x[0])
            return max(candidates, key=lambda x: x[0])
        def level_note(price, ref_label):
            return f"{ref_label} {format_usd(price)}"
        for i in range(max(BREAKOUT_LOOKBACK, VOLUME_LOOKBACK), len(price_df)):
            date = price_df.index[i]
            row = {
                "date": date,
                "open": o.iloc[i],
                "high": h.iloc[i],
                "low": l.iloc[i],
                "close": c.iloc[i],
                "volume": vol.iloc[i],
            }
            rh, rl = roll_high.iloc[i], roll_low.iloc[i]
            if pd.isna(rh) or pd.isna(rl):
                continue
            vr = float(vol_ratio.iloc[i]) if pd.notna(vol_ratio.iloc[i]) else 0.0
            vol_ok = vr >= VOLUME_CONFIRM_RATIO
            b, r = body.iloc[i], rng.iloc[i]
            bp = b / r if r > 0 else 0
            resist_price, resist_label = pick_resistance(c.iloc[i], rh)
            support_price, support_label = pick_support(c.iloc[i], rl)
            close_above = c.iloc[i] > resist_price
            body_break = min(o.iloc[i], c.iloc[i]) > resist_price * 0.998
            strong_candle = bullish.iloc[i] and bp >= 0.5
            if close_above and vol_ok and (body_break or strong_candle):
                note = (
                    f"Close {format_usd(c.iloc[i])} above {level_note(resist_price, resist_label)} "
                    f"with volume {vr:.1f}× avg ({VOLUME_CONFIRM_RATIO}× required)"
                )
                breakout_events.append({**row, "type": "Bullish Breakout", "signal": "Bullish", "level": resist_price, "note": note})
            close_below = c.iloc[i] < support_price
            body_break_dn = max(o.iloc[i], c.iloc[i]) < support_price * 1.002
            strong_bear = bearish.iloc[i] and bp >= 0.5
            if close_below and vol_ok and (body_break_dn or strong_bear):
                note = (
                    f"Close {format_usd(c.iloc[i])} below {level_note(support_price, support_label)} "
                    f"with volume {vr:.1f}× avg"
                )
                breakout_events.append({**row, "type": "Bearish Breakdown", "signal": "Bearish", "level": support_price, "note": note})
            if i < COMPRESS_BARS:
                continue
            recent_rng = rng.iloc[i - COMPRESS_BARS + 1 : i + 1]
            prior_rng = rng.iloc[i - 2 * COMPRESS_BARS + 1 : i - COMPRESS_BARS + 1]
            compressing = recent_rng.mean() < prior_rng.mean() * 0.85 if len(prior_rng) == COMPRESS_BARS else False
            nr7 = r <= rng.iloc[max(0, i - 6) : i + 1].min() * 1.001
            inside_bar = i >= 1 and h.iloc[i] <= h.iloc[i - 1] and l.iloc[i] >= l.iloc[i - 1]
            dist_resist = (resist_price - c.iloc[i]) / resist_price if resist_price else np.inf
            dist_support = (c.iloc[i] - support_price) / support_price if support_price else np.inf
            near_resist = 0 < dist_resist <= NEAR_LEVEL_PCT
            near_support = 0 < dist_support <= NEAR_LEVEL_PCT
            vol_rising = i >= 2 and vol.iloc[i] > vol.iloc[i - 1] > vol.iloc[i - 2]
            vol_dry_then_up = (
                i >= 3
                and vol.iloc[i - 2] < vol_avg.iloc[i - 2]
                and vol.iloc[i - 1] < vol_avg.iloc[i - 1]
                and vol.iloc[i] > vol.iloc[i - 1]
            )
            higher_lows = i >= 3 and l.iloc[i] > l.iloc[i - 1] > l.iloc[i - 2] and c.iloc[i] > c.iloc[i - 1]
            if near_resist and not close_above:
                hints = []
                if compressing:
                    hints.append("range compressing")
                if nr7:
                    hints.append("NR7 (narrowest range)")
                if inside_bar:
                    hints.append("inside bar")
                if vol_dry_then_up or (vol_rising and vr >= 1.1):
                    hints.append("volume building")
                if higher_lows:
                    hints.append("higher lows into level")
                if bullish.iloc[i] and bp >= 0.55:
                    hints.append("strong bullish candle")
                if len(hints) >= 2:
                    pre_breakout_events.append({
                        **row,
                        "type": "Pre-Breakout (Bullish)",
                        "signal": "Watch",
                        "level": resist_price,
                        "note": (
                            f"{format_usd(c.iloc[i])} within {dist_resist * 100:.1f}% of {level_note(resist_price, resist_label)}; "
                            + ", ".join(hints)
                        ),
                    })
            if near_support and not close_below:
                hints = []
                if compressing:
                    hints.append("range compressing")
                if nr7:
                    hints.append("NR7")
                if inside_bar:
                    hints.append("inside bar")
                if vol_dry_then_up or (vol_rising and vr >= 1.1):
                    hints.append("volume building")
                lower_highs = i >= 3 and h.iloc[i] < h.iloc[i - 1] < h.iloc[i - 2] and c.iloc[i] < c.iloc[i - 1]
                if lower_highs:
                    hints.append("lower highs into level")
                if bearish.iloc[i] and bp >= 0.55:
                    hints.append("strong bearish candle")
                if len(hints) >= 2:
                    pre_breakout_events.append({
                        **row,
                        "type": "Pre-Breakdown (Bearish)",
                        "signal": "Watch",
                        "level": support_price,
                        "note": (
                            f"{format_usd(c.iloc[i])} within {dist_support * 100:.1f}% of {level_note(support_price, support_label)}; "
                            + ", ".join(hints)
                        ),
                    })
        breakouts_df = pd.DataFrame(breakout_events)
        pre_breakouts_df = pd.DataFrame(pre_breakout_events)
        latest_bo = breakouts_df[breakouts_df["date"] == price_df.index[-1]] if not breakouts_df.empty else breakouts_df
        latest_pre = pre_breakouts_df[pre_breakouts_df["date"] == price_df.index[-1]] if not pre_breakouts_df.empty else pre_breakouts_df
        return breakouts_df, pre_breakouts_df, roll_high, roll_low, latest_bo, latest_pre, vol_ratio

    breakouts_df, pre_breakouts_df, roll_high, roll_low, latest_bo, latest_pre, _rolling_vol_ratio = detect_rolling_breakouts(df)

    def compute_macd(close: pd.Series) -> pd.DataFrame:
        ema_fast = close.ewm(span=MACD_FAST, adjust=False).mean()
        ema_slow = close.ewm(span=MACD_SLOW, adjust=False).mean()
        macd = ema_fast - ema_slow
        signal = macd.ewm(span=MACD_SIGNAL, adjust=False).mean()
        hist = macd - signal
        return pd.DataFrame({"macd": macd, "macd_signal": signal, "macd_hist": hist})

    def compute_adx(frame: pd.DataFrame, period: int = ADX_PERIOD) -> pd.DataFrame:
        high, low, close = frame["High"], frame["Low"], frame["Close"]
        up_move = high.diff()
        down_move = -low.diff()
        plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
        minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)
        prev_close = close.shift(1)
        tr = pd.concat(
            [high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1
        ).max(axis=1)
        atr = tr.ewm(alpha=1 / period, adjust=False).mean()
        plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
        minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
        di_sum = (plus_di + minus_di).replace(0, np.nan)
        dx = 100 * (plus_di - minus_di).abs() / di_sum
        adx = dx.ewm(alpha=1 / period, adjust=False).mean()
        return pd.DataFrame({"plus_di": plus_di, "minus_di": minus_di, "adx": adx})

    def adaptive_box(frame: pd.DataFrame, min_bars: int = CONSOL_MIN_BARS, max_bars: int = CONSOL_MAX_BARS) -> pd.DataFrame:
        """Find the tightest recent price range (manual box equivalent)."""
        box_high = pd.Series(np.nan, index=frame.index)
        box_low = pd.Series(np.nan, index=frame.index)
        box_width_pct = pd.Series(np.nan, index=frame.index)
        for i in range(max_bars, len(frame)):
            best_width = np.inf
            best_hi = best_lo = np.nan
            for w in range(min_bars, max_bars + 1):
                seg = frame.iloc[i - w : i]  # prior bars only (no lookahead)
                hi, lo = seg["High"].max(), seg["Low"].min()
                mid = (hi + lo) / 2
                if mid <= 0:
                    continue
                width = (hi - lo) / mid
                if width < best_width:
                    best_width, best_hi, best_lo = width, hi, lo
            box_high.iloc[i] = best_hi
            box_low.iloc[i] = best_lo
            box_width_pct.iloc[i] = best_width
        return pd.DataFrame({"box_high": box_high, "box_low": box_low, "box_width_pct": box_width_pct})

    def consecutive_true(mask: pd.Series) -> pd.Series:
        groups = (~mask).cumsum()
        return mask.groupby(groups).cumsum()
    # Build indicator frame
    work = df.copy()
    work["ema20"] = work["Close"].ewm(span=EMA_FAST, adjust=False).mean()
    work["ema50"] = work["Close"].ewm(span=EMA_MID, adjust=False).mean()
    work["ema200"] = work["Close"].ewm(span=EMA_SLOW, adjust=False).mean()
    work["vol_ma"] = work["Volume"].rolling(VOL_MA, min_periods=5).mean()
    work["vol_ratio"] = work["Volume"] / work["vol_ma"].replace(0, np.nan)
    work = work.join(compute_macd(work["Close"]))
    work = work.join(compute_adx(work))
    work = work.join(adaptive_box(work))
    work["consolidating"] = work["box_width_pct"] <= BOX_RANGE_PCT_MAX
    work["vol_quiet"] = work["Volume"] < work["vol_ma"] * VOL_QUIET_RATIO
    work["vol_quiet_run"] = consecutive_true(work["vol_quiet"].fillna(False))
    work["vol_quiet_zone"] = work["vol_quiet_run"] >= VOL_QUIET_MIN_BARS
    work["ema_bull_stack"] = (
        (work["Close"] > work["ema20"])
        & (work["ema20"] > work["ema50"])
        & (work["ema50"] > work["ema200"])
    )
    work["near_box_top"] = (
        work["consolidating"]
        & ((work["box_high"] - work["Close"]) / work["box_high"].replace(0, np.nan) <= NEAR_BOX_TOP_PCT)
        & (work["Close"] >= work["box_low"])
    )
    # MACD "duration" reset: negative histogram cooling off, trending toward zero
    neg_hist = work["macd_hist"] < 0
    neg_run = consecutive_true(neg_hist.fillna(False))
    hist_improving = work["macd_hist"] > work["macd_hist"].shift(1)
    work["macd_duration_reset"] = (neg_run >= MACD_DURATION_MIN) & hist_improving
    work["adx_rising"] = work["adx"] > work["adx"].shift(1)
    work["adx_curl_up"] = (work["adx"] < 30) & work["adx_rising"]
    rng = work["High"] - work["Low"]
    work["range_narrow"] = rng < rng.shift(1).rolling(5, min_periods=3).mean() * 0.9
    work["breakout_close"] = (
        work["Close"] > work["box_high"]
    ) & (work["vol_ratio"] >= VOLUME_CONFIRM_RATIO)
    work["pre_breakout"] = False
    work["pre_score"] = 0
    checks = {
        "consolidating": work["consolidating"],
        "near_box_top": work["near_box_top"],
        "vol_quiet_zone": work["vol_quiet_zone"] & work["consolidating"],
        "ema_bull_stack": work["ema_bull_stack"],
        "macd_duration_reset": work["macd_duration_reset"] & work["consolidating"],
        "adx_curl_up": work["adx_curl_up"],
    }
    for name, series in checks.items():
        work["pre_score"] += series.astype(int)
    not_broken = work["Close"] <= work["box_high"] * 1.002
    work.loc[not_broken & (work["pre_score"] >= PRE_BREAKOUT_MIN_SCORE), "pre_breakout"] = True
    # Event log
    events = []
    for i in range(CONSOL_MAX_BARS, len(work)):
        row = work.iloc[i]
        if row["breakout_close"]:
            events.append({
                "date": work.index[i],
                "type": "Breakout (confirmed)",
                "close": row["Close"],
                "level": row["box_high"],
                "vol_ratio": row["vol_ratio"],
                "note": f"Close {format_usd(row['Close'])} above box {format_usd(row['box_high'])} with {row['vol_ratio']:.1f}× avg volume",
            })
        elif row["pre_breakout"]:
            hints = [k.replace("_", " ") for k, s in checks.items() if bool(s.iloc[i])]
            events.append({
                "date": work.index[i],
                "type": "Pre-breakout watch",
                "close": row["Close"],
                "level": row["box_high"],
                "vol_ratio": row["vol_ratio"],
                "note": f"{format_usd(row['Close'])} near box top {format_usd(row['box_high'])}; score {int(row['pre_score'])}/6 — {', '.join(hints)}",
            })
    events_df = pd.DataFrame(events)
    # Confirmed breakout/breakdown from either system wins: drop pre-breakout on those dates
    confirmed_dates = set()
    if not breakouts_df.empty:
        confirmed_dates.update(pd.to_datetime(breakouts_df["date"]))
    if not events_df.empty:
        confirmed_dates.update(
            pd.to_datetime(events_df.loc[events_df["type"] == "Breakout (confirmed)", "date"])
        )
    confirmed_dates.update(work.index[work["breakout_close"].fillna(False)])
    if not pre_breakouts_df.empty:
        pre_breakouts_df = pre_breakouts_df.loc[
            ~pd.to_datetime(pre_breakouts_df["date"]).isin(confirmed_dates)
        ].reset_index(drop=True)
    if not events_df.empty:
        events_df = events_df.loc[
            ~((events_df["type"] == "Pre-breakout watch") & pd.to_datetime(events_df["date"]).isin(confirmed_dates))
        ].reset_index(drop=True)
    work.loc[work.index.isin(confirmed_dates), "pre_breakout"] = False
    latest_bo = (
        breakouts_df[breakouts_df["date"] == work.index[-1]] if not breakouts_df.empty else breakouts_df
    )
    latest_pre = (
        pre_breakouts_df[pre_breakouts_df["date"] == work.index[-1]] if not pre_breakouts_df.empty else pre_breakouts_df
    )
    latest = work.iloc[-1]
    # Latest status
    if latest["breakout_close"]:
        status = f"BREAKOUT — close above box with {latest['vol_ratio']:.1f}× volume"
    elif latest["pre_breakout"]:
        status = f"PRE-BREAKOUT WATCH — score {int(latest['pre_score'])}/6 near {format_usd(latest['box_high'])}"
    elif latest["consolidating"]:
        status = f"Consolidating in box {format_usd(latest['box_low'])}–{format_usd(latest['box_high'])}"
    else:
        status = "No active consolidation setup"
    if not latest_bo.empty:
        v2_status = latest_bo.iloc[-1]["note"]
    elif not latest_pre.empty:
        v2_status = latest_pre.iloc[-1]["note"]
    else:
        v2_status = (
            f"No rolling breakout — need close above {format_usd(roll_high.iloc[-1])} "
            f"with ≥{VOLUME_CONFIRM_RATIO}× avg volume"
        )
    summary = [
        f"Latest signal (box): {status}",
        f"Latest signal (rolling): {v2_status}",
    ]
    breakouts_table, pre_breakouts_table = build_setup_tables(
        work, breakouts_df, pre_breakouts_df, events_df
    )
    # --- Chart window ---
    plot_df = work.iloc[-chart_days:].copy()
    box = plot_df[["box_high", "box_low", "consolidating"]].dropna()
    if not box.empty:
        active = box[box["consolidating"]]
        if not active.empty:
            last_box = active.iloc[-1]
            box_high, box_low = float(last_box["box_high"]), float(last_box["box_low"])
            box_start = active.index[0]
        else:
            box_high = box_low = box_start = None
    else:
        box_high = box_low = box_start = None
    macd_hist_colors = np.where(plot_df["macd_hist"] >= 0, _UP, _DOWN)
    apds = [
        mpf.make_addplot(plot_df["ema20"], color=_ACCENT, width=1.1),
        mpf.make_addplot(plot_df["ema50"], color=_MID, width=1.1),
        mpf.make_addplot(plot_df["ema200"], color="#7E57C2", width=1.1),
        mpf.make_addplot(plot_df["vol_ma"], panel=1, color=_TEXT_MUTED, width=1.0, linestyle="--"),
        mpf.make_addplot(plot_df["macd"], panel=2, color=_MID, width=1.0),
        mpf.make_addplot(plot_df["macd_signal"], panel=2, color="#FF9800", width=1.0),
        mpf.make_addplot(
            plot_df["macd_hist"], panel=2, type="bar", color=macd_hist_colors, width=0.7, alpha=0.85
        ),
        mpf.make_addplot(plot_df["adx"], panel=3, color="#7E57C2", width=1.2),
        mpf.make_addplot(plot_df["plus_di"], panel=3, color=_UP, width=0.8),
        mpf.make_addplot(plot_df["minus_di"], panel=3, color=_DOWN, width=0.8),
    ]
    # Rolling high/low levels
    apds.extend([
        mpf.make_addplot(roll_high.reindex(plot_df.index), color=_DOWN, width=0.9, linestyle=":"),
        mpf.make_addplot(roll_low.reindex(plot_df.index), color=_UP, width=0.9, linestyle=":"),
    ])
    # Canonical price markers: breakout, breakdown, and bullish pre-breakout only
    breakout_pts = pd.Series(np.nan, index=plot_df.index)
    breakdown_pts = pd.Series(np.nan, index=plot_df.index)
    pre_breakout_pts = pd.Series(np.nan, index=plot_df.index)
    if not breakouts_df.empty:
        chart_breakouts = breakouts_df[breakouts_df["date"].isin(plot_df.index)]
        for _, hit in chart_breakouts.iterrows():
            if hit["signal"] == "Bullish":
                breakout_pts.loc[hit["date"]] = hit["high"] * 1.04
            elif hit["signal"] == "Bearish":
                breakdown_pts.loc[hit["date"]] = hit["low"] * 0.96
    box_breakouts = plot_df.index[plot_df["breakout_close"].fillna(False)]
    for idx in box_breakouts:
        breakout_pts.loc[idx] = plot_df.loc[idx, "High"] * 1.04
    if not pre_breakouts_df.empty:
        chart_pre_breakouts = pre_breakouts_df[
            pre_breakouts_df["date"].isin(plot_df.index)
            & (pre_breakouts_df["type"] == "Pre-Breakout (Bullish)")
            & ~pre_breakouts_df["date"].isin(breakout_pts.dropna().index)
            & ~pre_breakouts_df["date"].isin(breakdown_pts.dropna().index)
        ]
        for _, hit in chart_pre_breakouts.iterrows():
            pre_breakout_pts.loc[hit["date"]] = hit["high"] * 1.02
    if breakout_pts.notna().any():
        apds.append(mpf.make_addplot(
            breakout_pts, type="scatter", markersize=110, marker="^", color=_UP
        ))
    if breakdown_pts.notna().any():
        apds.append(mpf.make_addplot(
            breakdown_pts, type="scatter", markersize=110, marker="v", color=_DOWN
        ))
    if pre_breakout_pts.notna().any():
        apds.append(mpf.make_addplot(
            pre_breakout_pts, type="scatter", markersize=90, marker="D", color=_ACCENT
        ))
    fig, axes = mpf.plot(
        plot_df,
        type="candle",
        style=_STYLE,
        volume=True,
        addplot=apds,
        panel_ratios=(4, 1, 1, 1),
        figsize=(16, 11),
        title=f"{pair} — Breakout Analysis ({chart_days}d)",
        ylabel="Price (USDT)",
        returnfig=True,
        warn_too_much_data=10000,
        datetime_format="%b %d",
    )
    fig.patch.set_facecolor(_BG)
    _style_axes(axes)
    ax_price = axes[0]
    ax_vol = axes[2]
    ax_macd = axes[4]
    ax_adx = axes[6]

    def bar_x(frame: pd.DataFrame, ts) -> int:
        """Map session date to mplfinance integer bar index (not matplotlib date)."""
        loc = frame.index.get_loc(ts)
        return int(loc[0] if isinstance(loc, slice) else loc)
    x_end = len(plot_df) - 1
    # Consolidation box shading (latest active box in window)
    if box_high is not None and box_start is not None:
        x0 = bar_x(plot_df, box_start) if box_start in plot_df.index else 0
        xmin = max(0, x0) / max(x_end, 1)
        ax_price.axhspan(
            box_low,
            box_high,
            xmin=xmin,
            xmax=1.0,
            color=_MID,
            alpha=0.12,
            zorder=0,
        )
        ax_price.axhline(box_high, color=_DOWN, linestyle="--", linewidth=0.9, alpha=0.7)
        ax_price.axhline(box_low, color=_UP, linestyle="--", linewidth=0.9, alpha=0.7)
        ax_price.text(
            x_end,
            box_high,
            f" box top {format_usd(box_high)}",
            color=_DOWN,
            fontsize=8,
            va="bottom",
        )
    # Volume quiet / drying up annotations
    quiet = plot_df[plot_df["vol_quiet_zone"]]
    if not quiet.empty:
        clusters = []
        run_start = None
        prev = None
        for idx in quiet.index:
            if run_start is None:
                run_start = idx
            elif prev is not None and (bar_x(plot_df, idx) - bar_x(plot_df, prev)) > 1:
                clusters.append((run_start, prev))
                run_start = idx
            prev = idx
        if run_start is not None and prev is not None:
            clusters.append((run_start, prev))
        # headroom so the callout box is not clipped by the volume panel
        ax_vol.set_ylim(top=float(plot_df["Volume"].max()) * 1.35)
        for start, end in clusters[-2:]:
            seg = plot_df.loc[start:end]
            mid = seg.index[len(seg) // 2]
            mid_x = bar_x(plot_df, mid)
            vol_y = float(plot_df.loc[mid, "Volume"])
            vol_top = float(plot_df["Volume"].max())
            ax_vol.annotate(
                "Volume quiet",
                xy=(mid_x, vol_y),
                xytext=(mid_x, max(vol_y * 1.2, vol_top * 0.55)),
                fontsize=8,
                color=_TEXT,
                ha="center",
                va="bottom",
                zorder=6,
                bbox=dict(
                    boxstyle="round,pad=0.35",
                    facecolor=_PANEL,
                    edgecolor=_BORDER,
                    linewidth=1.2,
                    alpha=0.95,
                ),
                arrowprops=dict(arrowstyle="-|>", color=_TEXT_MUTED, lw=1.6, shrinkB=2),
            )
    # MACD duration marker (latest negative-hist reset segment in window)
    macd_seg = plot_df[plot_df["macd_duration_reset"]]
    if not macd_seg.empty:
        seg_start = macd_seg.index[0]
        seg_end = macd_seg.index[-1]
        seg_start_x = bar_x(plot_df, seg_start)
        seg_end_x = bar_x(plot_df, seg_end)
        hist_start = float(plot_df.loc[seg_start, "macd_hist"])
        hist_end = float(plot_df.loc[seg_end, "macd_hist"])
        ax_macd.annotate(
            "",
            xy=(seg_end_x, hist_end),
            xytext=(seg_start_x, hist_start),
            arrowprops=dict(arrowstyle="-", color=_TEXT_MUTED, lw=1.2),
        )
        ax_macd.text(
            seg_start_x,
            float(plot_df.loc[seg_start:seg_end, "macd_hist"].min()) * 1.2,
            "duration",
            color=_TEXT_MUTED,
            fontsize=8,
        )
    ema_legend = [
        Line2D([0], [0], color=_ACCENT, linewidth=1.4, label="EMA 20"),
        Line2D([0], [0], color=_MID, linewidth=1.4, label="EMA 50"),
        Line2D([0], [0], color="#7E57C2", linewidth=1.4, label="EMA 200"),
        Line2D([0], [0], color=_DOWN, linewidth=1.2, linestyle=":", label="20d high"),
        Line2D([0], [0], color=_UP, linewidth=1.2, linestyle=":", label="20d low"),
        Line2D([0], [0], marker="^", color=_UP, linestyle="None", markersize=7, label="Breakout"),
        Line2D([0], [0], marker="v", color=_DOWN, linestyle="None", markersize=7, label="Breakdown"),
        Line2D([0], [0], marker="D", color=_ACCENT, linestyle="None", markersize=6, label="Pre-breakout"),
    ]
    ax_price.legend(
        handles=ema_legend,
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
        facecolor=_PANEL,
        edgecolor=_BORDER,
        labelcolor=_TEXT,
    )
    ax_macd.set_ylabel("MACD")
    ax_adx.set_ylabel("ADX")
    ax_price.set_title("")
    fig.subplots_adjust(top=0.84, hspace=0.05)
    fig.suptitle(
        f"{pair} — Breakout Analysis ({chart_days}d)",
        color=_TEXT,
        fontsize=14,
        fontweight="bold",
        y=0.98,
    )
    for index, line in enumerate(summary):
        fig.text(
            0.5,
            0.935 - index * 0.028,
            line,
            ha="center",
            va="top",
            color=_TEXT_MUTED,
            fontsize=9,
        )
    return BreakoutChart(
        figure=fig,
        summary=summary,
        breakouts=breakouts_table,
        pre_breakouts=pre_breakouts_table,
    )

if __name__ == "__main__":
    result = create_breakout_plot("BTC/USDT")
    plt.show()
