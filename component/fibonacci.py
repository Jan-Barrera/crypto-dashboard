import matplotlib.pyplot as plt
import streamlit as st

from config import load_crypto_pairs, to_display_pair
from db.hist_data import get_hist_data
from model.fibonacci import plot_fibonacci_retracement
from ui.dashboard import render_footer


def _pairs() -> list[str]:
    return [to_display_pair(pair) for pair in load_crypto_pairs()]


def _render_fibonacci_chart(ticker: str, *, show_spinner: bool = False) -> bool:
    try:
        if show_spinner:
            with st.spinner(f"Loading {ticker} price data..."):
                df = get_hist_data(ticker)
                result = plot_fibonacci_retracement(ticker, df)
        else:
            df = get_hist_data(ticker)
            result = plot_fibonacci_retracement(ticker, df)
        st.pyplot(result.figure, width="stretch")
        plt.close(result.figure)
        return True
    except Exception as exc:
        st.error(f"Could not generate Fibonacci chart for {ticker}: {exc}")
        return False


def render_fibonacci_retracement() -> None:
    pairs = _pairs()
    default_pair = st.session_state.get("fibonacci_active_ticker") or pairs[0]
    st.session_state.setdefault(
        "fibonacci_symbol",
        default_pair if default_pair in pairs else pairs[0],
    )
    active_ticker = st.session_state.get("fibonacci_active_ticker")

    st.write("")
    st.markdown(
        """
        <div class="section-title">📐 Fibonacci Retracement</div>
        <div class="section-sub">Find potential support and resistance levels</div>
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
        </style>
        """,
        unsafe_allow_html=True,
    )

    label_col, form_col = st.columns([1, 4], vertical_alignment="center", width=450)

    with label_col:
        st.markdown("**Pair:**")

    with form_col:
        with st.form("fibonacci_form", clear_on_submit=False, border=False):
            input_col, button_col = st.columns([3, 1], vertical_alignment="center", gap="small")
            with input_col:
                symbol = st.selectbox(
                    "symbol",
                    pairs,
                    index=pairs.index(st.session_state.fibonacci_symbol)
                    if st.session_state.fibonacci_symbol in pairs
                    else 0,
                    label_visibility="collapsed",
                )
            with button_col:
                generate = st.form_submit_button("Generate", type="secondary", width="stretch")

    if generate:
        st.session_state.fibonacci_symbol = symbol
        ticker = st.session_state.fibonacci_symbol
        if _render_fibonacci_chart(ticker, show_spinner=True):
            st.session_state.fibonacci_active_ticker = ticker
    elif active_ticker:
        _render_fibonacci_chart(active_ticker)

    st.write("")
    render_footer()
