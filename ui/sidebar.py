import streamlit as st


def get_nav_items() -> list[tuple[str, str]]:
    return [
        ("📊", "Market Overview"),
        ("⭐", "Swing Trade Watchlist"),
        ("📐", "Fibonacci Retracement"),
        ("🐂", "Breakout Setup"),
    ]


def render_sidebar() -> str:
    nav_items = get_nav_items()
    nav_options = [f"{icon}  {label}" for icon, label in nav_items]

    with st.sidebar:
        st.markdown(
            """
            <div style="display:flex;align-items:center;gap:10px;padding:6px 0 20px 0;">
                <div style="font-size:26px;">₿</div>
                <div style="font-size:20px;font-weight:800;color:#fff;">Crypto Dashboard</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        selected = st.radio(
            "nav",
            nav_options,
            index=0,
            label_visibility="collapsed",
            key="main_nav",
        )

        st.markdown("<hr style='border-color:#1c2333;margin:20px 0;'>", unsafe_allow_html=True)

    for icon, label in nav_items:
        if selected == f"{icon}  {label}":
            return label
    return nav_items[0][1]
