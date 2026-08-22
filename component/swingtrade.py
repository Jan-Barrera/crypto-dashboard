import streamlit as st

from model.swingtrade import get_swing_trade_watchlist
from ui.dashboard import render_footer

_ROW_HEIGHT = 35
_HEADER_HEIGHT = 38


def _table_height(row_count: int) -> int:
    return _HEADER_HEIGHT + max(row_count, 1) * _ROW_HEIGHT


@st.cache_data(ttl=900, show_spinner=False)
def _load_watchlist():
    return get_swing_trade_watchlist()


def render_swing_trade_watchlist() -> None:
    st.write("")
    st.markdown(
        """
        <div class="section-title">⭐ SWING TRADE WATCHLIST</div>
        <div class="section-sub">Bullish filters on daily candles</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    with st.spinner("Scanning swing trade setups..."):
        watchlist = _load_watchlist()
    if watchlist.empty:
        st.info("No pairs currently pass the bullish filters.")
        render_footer()
        return
    st.markdown(
        """
        <style>
            [data-testid="stDataFrame"] > div {
                overflow: hidden !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.dataframe(
        watchlist,
        width="stretch",
        height=_table_height(len(watchlist)),
        row_height=_ROW_HEIGHT,
        hide_index=True,
        column_config={
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Name": st.column_config.TextColumn("Name"),
            "Date": st.column_config.TextColumn("Date"),
            "Close": st.column_config.TextColumn("Close"),
            "Support": st.column_config.TextColumn("Support"),
            "Resistance": st.column_config.TextColumn("Resistance"),
            "Stop Loss": st.column_config.TextColumn("Stop Loss"),
            "Take Profit": st.column_config.TextColumn("Take Profit"),
            "RSI": st.column_config.TextColumn("RSI"),
            "ADX": st.column_config.TextColumn("ADX"),
        },
    )
    render_footer()
