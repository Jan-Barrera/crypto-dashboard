import matplotlib.pyplot as plt
import streamlit as st

from config import load_crypto_pairs, to_display_pair
from model.breakout import create_breakout_plot
from ui.dashboard import render_footer

_ROW_HEIGHT = 35
_HEADER_HEIGHT = 38

_SETUP_COLUMN_CONFIG = {
    "Date": st.column_config.TextColumn("Date"),
    "Type": st.column_config.TextColumn("Type"),
    "Signal": st.column_config.TextColumn("Signal"),
    "Close": st.column_config.TextColumn("Close"),
    "Level": st.column_config.TextColumn("Level"),
    "Note": st.column_config.TextColumn("Note"),
}


def _pairs() -> list[str]:
    return [to_display_pair(pair) for pair in load_crypto_pairs()]


def _table_height(row_count: int) -> int:
    return _HEADER_HEIGHT + max(row_count, 1) * _ROW_HEIGHT


def _render_setup_table(title: str, df) -> None:
    st.markdown(f"**{title}**")
    if df.empty:
        st.caption("No setups in the last 90 days.")
        return
    st.dataframe(
        df,
        width="stretch",
        height=_table_height(len(df)),
        row_height=_ROW_HEIGHT,
        hide_index=True,
        column_config=_SETUP_COLUMN_CONFIG,
    )


def _render_breakout_chart(ticker: str, *, show_spinner: bool = False) -> bool:
    try:
        if show_spinner:
            with st.spinner(f"Loading {ticker} breakout chart..."):
                result = create_breakout_plot(ticker)
        else:
            result = create_breakout_plot(ticker)
        st.pyplot(result.figure, width="stretch")
        plt.close(result.figure)
        st.write("")
        _render_setup_table("Confirmed breakouts (last 90 days)", result.breakouts)
        st.write("")
        _render_setup_table("Pre-breakout / pre-breakdown setups (last 90 days)", result.pre_breakouts)
        return True
    except Exception as exc:
        st.error(f"Could not generate breakout chart for {ticker}: {exc}")
        return False


def render_breakout_watchlist() -> None:
    pairs = _pairs()
    default_pair = st.session_state.get("breakout_active_ticker") or pairs[0]
    st.session_state.setdefault(
        "breakout_symbol",
        default_pair if default_pair in pairs else pairs[0],
    )
    active_ticker = st.session_state.get("breakout_active_ticker")

    st.write("")
    st.markdown(
        """
        <div class="section-title">🐂 BREAKOUT WATCHLIST</div>
        <div class="section-sub">Breakouts and pre-breakout setups</div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")
    st.markdown(
        """
        <style>
            div[data-testid="InputInstructions"] { visibility: hidden; }
            div[data-testid="stForm"] [data-testid="column"] {
                display: flex;
                align-items: center;
            }
            div[data-testid="stForm"] [data-testid="stSelectbox"] {
                width: 100%;
            }
            div[data-testid="stForm"] [data-testid="stFormSubmitButton"] > button,
            div[data-testid="stForm"] [data-testid="stButton"] > button {
                height: 2.5rem;
                min-height: 2.5rem;
                margin: 0;
                box-sizing: border-box;
            }
            [data-testid="stDataFrame"] > div {
                overflow: hidden !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    label_col, form_col = st.columns([1, 4], vertical_alignment="center", width=450)

    with label_col:
        st.markdown("**Pair:**")

    with form_col:
        with st.form("breakout_form", clear_on_submit=False, border=False):
            input_col, button_col = st.columns([3, 1], vertical_alignment="center", gap="small")
            with input_col:
                symbol = st.selectbox(
                    "symbol",
                    pairs,
                    index=pairs.index(st.session_state.breakout_symbol)
                    if st.session_state.breakout_symbol in pairs
                    else 0,
                    label_visibility="collapsed",
                )
            with button_col:
                generate = st.form_submit_button("Generate", type="secondary", width="stretch")

    if generate:
        st.session_state.breakout_symbol = symbol
        ticker = st.session_state.breakout_symbol
        if _render_breakout_chart(ticker, show_spinner=True):
            st.session_state.breakout_active_ticker = ticker
    elif active_ticker:
        _render_breakout_chart(active_ticker)

    st.write("")
    render_footer()
