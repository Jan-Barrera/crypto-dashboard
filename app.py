import streamlit as st

from ui.dashboard import render_dashboard
from ui.sidebar import render_sidebar
from ui.style import apply_global_styles
from component.swingtrade import render_swing_trade_watchlist
from component.fibonacci import render_fibonacci_retracement
from component.breakout import render_breakout_watchlist


st.set_page_config(
    page_title="Crypto Market Dashboard",
    page_icon="₿",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_global_styles()
selected_page = render_sidebar()

if selected_page == "Market Overview":
    render_dashboard()
elif selected_page == "Swing Trade Watchlist":
    render_swing_trade_watchlist()
elif selected_page == "Fibonacci Retracement":
    render_fibonacci_retracement()
elif selected_page == "Breakout Setup":
    render_breakout_watchlist()
else:
    render_dashboard()
