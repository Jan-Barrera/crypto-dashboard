import logging
from dataclasses import dataclass

import matplotlib as mpl
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import mplfinance as mpf
import pandas as pd

from config import LOOKBACK_DAYS, format_usd, to_display_pair

pd.set_option("display.max_colwidth", None)
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.weight": "normal",
    "axes.labelweight": "bold",
    "figure.titleweight": "bold",
})
logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

# Match ui/style.css dark palette
_BG = "#0b0f19"
_PANEL = "#11172a"
_BORDER = "#1e2540"
_TEXT = "#e6e9ef"
_TEXT_MUTED = "#8b93a7"
_UP = "#26d07c"
_DOWN = "#f0555e"

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

_REQUIRED_COLS = ["Open", "High", "Low", "Close", "Volume"]
_FIB_RATIOS = {
    "0.0%": 0.0,
    "23.6%": 0.236,
    "38.2%": 0.382,
    "50.0%": 0.5,
    "61.8%": 0.618,
    "78.6%": 0.786,
    "100.0%": 1.0,
}


@dataclass
class FibonacciChart:
    figure: plt.Figure
    summary: list[str]


def plot_fibonacci_retracement(
    symbol: str,
    df: pd.DataFrame,
    lookback_days: int = LOOKBACK_DAYS,
) -> FibonacciChart:
    """Build a candlestick chart with Fibonacci retracement levels."""
    missing = [name for name in _REQUIRED_COLS if name not in df.columns]
    if missing:
        raise KeyError(f"Missing OHLCV columns {missing}; got {list(df.columns)}")
    if df.index.name != "date":
        raise ValueError("Expected df indexed by date from Binance klines")

    window = df[df.index >= df.index[-1] - pd.Timedelta(days=lookback_days)]
    swing_high = window["High"].max()
    swing_low = window["Low"].min()
    high_date = window["High"].idxmax()
    low_date = window["Low"].idxmin()
    price_range = abs(swing_high - swing_low)
    direction = "down" if high_date < low_date else "up"
    trend = "Downtrend" if direction == "down" else "Uptrend"

    if direction == "down":
        levels = {
            label: swing_low + ratio * price_range
            for label, ratio in _FIB_RATIOS.items()
        }
    else:
        levels = {
            label: swing_high - ratio * price_range
            for label, ratio in _FIB_RATIOS.items()
        }

    latest_close = df["Close"].iloc[-1]
    supports = [(label, price) for label, price in levels.items() if price <= latest_close]
    resistances = [(label, price) for label, price in levels.items() if price >= latest_close]

    hlines_prices = []
    hlines_colors = []
    hlines_widths = []
    nearest_support = max(supports, key=lambda x: x[1]) if supports else None
    nearest_resistance = min(resistances, key=lambda x: x[1]) if resistances else None

    for label, price in levels.items():
        hlines_prices.append(price)
        if price <= latest_close:
            hlines_colors.append(_UP)
            hlines_widths.append(
                2.0 if nearest_support and price == nearest_support[1] else 1.0
            )
        else:
            hlines_colors.append(_DOWN)
            hlines_widths.append(
                2.0 if nearest_resistance and price == nearest_resistance[1] else 1.0
            )

    pair = to_display_pair(symbol)
    fig, axes = mpf.plot(
        df,
        type="candle",
        style=_STYLE,
        title=f"{pair} - Fibonacci Retracement — {trend} ({lookback_days}d swing)",
        ylabel="Price (USDT)",
        volume=True,
        hlines=dict(
            hlines=hlines_prices,
            colors=hlines_colors,
            linestyle="--",
            linewidths=hlines_widths,
        ),
        figsize=(14, 8),
        warn_too_much_data=1000,
        datetime_format="%b %d",
        returnfig=True,
    )

    price_ax = axes[0]
    for (label, price), color in zip(levels.items(), hlines_colors):
        level_type = "S" if price <= latest_close else "R"
        price_ax.text(
            0.005,
            price,
            f" {level_type} {label}  {format_usd(price)} ",
            transform=price_ax.get_yaxis_transform(),
            ha="left",
            va="center",
            color=color,
            fontsize=9,
            fontweight="bold",
            bbox=dict(facecolor=_PANEL, edgecolor=color, alpha=0.95, pad=2),
            clip_on=True,
        )

    fig.patch.set_facecolor(_BG)
    date_fmt = mdates.DateFormatter("%b %d")
    for ax in axes:
        ax.set_facecolor(_PANEL)
        for spine in ax.spines.values():
            spine.set_color(_BORDER)
        ax.tick_params(colors=_TEXT_MUTED)
        ax.tick_params(axis="x", labelrotation=0)
        ax.yaxis.label.set_color(_TEXT)
        ax.xaxis.label.set_color(_TEXT)
        ax.title.set_color(_TEXT)
        ax.xaxis.set_major_formatter(date_fmt)
        for label in ax.get_xticklabels():
            label.set_rotation(0)
            label.set_ha("center")

    axes[-1].xaxis.set_major_locator(mdates.AutoDateLocator(minticks=6, maxticks=10))

    support_text = (
        f"{nearest_support[0]} at {format_usd(nearest_support[1])}"
        if nearest_support
        else "n/a"
    )
    resistance_text = (
        f"{nearest_resistance[0]} at {format_usd(nearest_resistance[1])}"
        if nearest_resistance
        else "n/a"
    )
    summary = [
        (
            f"Close: {format_usd(latest_close)}  |  "
            f"Swing: {format_usd(swing_low)} → {format_usd(swing_high)} ({direction}) | "
            f"Nearest support: {support_text} | "
            f"Nearest resistance: {resistance_text}"
        ),
    ]

    price_ax.set_title("")
    fig.subplots_adjust(top=0.84)
    fig.suptitle(
        f"{pair} - Fibonacci Retracement — {trend} ({lookback_days}d swing)",
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

    return FibonacciChart(figure=fig, summary=summary)
