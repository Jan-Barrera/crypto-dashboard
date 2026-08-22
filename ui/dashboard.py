from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st

from model.scrape_data import (
    SCRAPE_INTERVAL_SECONDS,
    get_checklist_left,
    get_checklist_right,
    get_indices,
    get_indices_updated_at,
    get_market_breadth,
    get_market_news,
    get_watchlist_data,
    indices_needs_refresh,
)

DASHBOARD_REFRESH = timedelta(seconds=SCRAPE_INTERVAL_SECONDS)


def render_header() -> None:
    top_l, top_r = st.columns([3, 2])
    with top_l:
        st.markdown(
            """
            <div class="top-bar-title">CRYPTO MARKET</div>
            """,
            unsafe_allow_html=True,
        )
    with top_r:
        now = datetime.now(ZoneInfo("Asia/Manila"))
        date_label = (
            f"{now.strftime('%B')} {now.day}, {now.strftime('%Y')}"
            f" &nbsp;•&nbsp; {now.strftime('%A')}"
        )
        st.markdown(
            f"""
            <div style="display:flex;justify-content:flex-end;align-items:center;gap:18px;padding-top:8px;">
                <span style="color:#c7cce0;font-size:13px;">{date_label}</span>
                <span class="market-status-open">● Market Open 24/7</span>
            </div>
            """,
            unsafe_allow_html=True,
        )


def _draw_index_cards(indices: list[dict]) -> None:
    updated_at = get_indices_updated_at()
    if updated_at is not None:
        st.session_state["indices_updated_at"] = updated_at

    breadth = get_market_breadth()
    total = max(breadth["total"], 1)
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1.1])
    cols_map = [c1, c2, c3, c4]
    for col, idx in zip(cols_map, indices):
        change_class = "card-change-up" if idx["up"] else "card-change-down"
        change_arrow = "▲" if idx["up"] else "▼"
        with col:
            st.markdown(
                f"""
                <div class="card">
                    <div class="card-label">{idx['name']}</div>
                    <div class="card-value">{idx['value']}</div>
                    <div class="{change_class}">{change_arrow} {idx['change']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            if idx["value_b"]:
                st.markdown(
                    f"""
                    <div class="sub-stats">
                        <div><div class="sub-stat-label">Volume</div><div class="sub-stat-value">{idx['value_b']}</div></div>
                        <div><div class="sub-stat-label">High</div><div class="sub-stat-value">{idx['vol']}</div></div>
                        <div><div class="sub-stat-label">Low</div><div class="sub-stat-value">{idx['trades']}</div></div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    with c5:
        st.markdown(
            f"""
            <div class="card" style="height: 100%;">
                <div class="card-label">Watchlist Breadth</div>
                <div style="display:flex;align-items:center;gap:16px;margin-top:8px;">
                    <div style="font-size:40px;">🍩</div>
                    <div style="font-size:13px;">
                        <div style="margin-bottom:4px;"><span style="color:#26d07c;">●</span> {breadth['advancers']} Advancers ({breadth['advancers']*100//total}%)</div>
                        <div style="margin-bottom:4px;"><span style="color:#f0555e;">●</span> {breadth['decliners']} Decliners ({breadth['decliners']*100//total}%)</div>
                        <div><span style="color:#9aa2b8;">●</span> {breadth['unchanged']} Unchanged ({breadth['unchanged']*100//total}%)</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_watchlist_header() -> None:
    st.markdown(
        """
        <div class="section-title">📊 MARKET WATCHLIST</div>
        """,
        unsafe_allow_html=True,
    )


def _watchlist_row_styles(row: pd.Series) -> list[str]:
    change = str(row.get("% Change", ""))
    if change.startswith("-"):
        style = "background-color: rgba(240, 85, 94, 0.18); color: #f0555e;"
    elif change.startswith("+"):
        style = "background-color: rgba(38, 208, 124, 0.18); color: #26d07c;"
    else:
        style = ""
    return [style] * len(row)


def render_watchlist_table(df: pd.DataFrame) -> None:
    if df.empty:
        st.caption("No watchlist data available.")
        return

    styled = df.style.apply(_watchlist_row_styles, axis=1)
    st.dataframe(
        styled,
        width="stretch",
        hide_index=True,
        column_config={
            "#": st.column_config.TextColumn("#"),
            "Symbol": st.column_config.TextColumn("Symbol"),
            "Name": st.column_config.TextColumn("Name"),
            "Last Price": st.column_config.TextColumn("Last Price"),
            "% Change": st.column_config.TextColumn("% Change"),
            "24h High": st.column_config.TextColumn("24h High"),
            "24h Low": st.column_config.TextColumn("24h Low"),
            "Volume": st.column_config.TextColumn("Volume"),
        },
    )


def _draw_market_news(news: list[tuple[str, str, str]]) -> None:
    news_html = """<div class='card'><div style="display:flex;justify-content:space-between;align-items:center;">
    <div class='section-title' style='font-size:15px;'>TOP MOVERS (24H)</div>
    <div class='section-sub' style='font-size:12px;'>Binance</div>
    </div><div style='margin-top:12px;'>"""
    if news:
        for change, headline, _url in news:
            change_color = "#26d07c" if change.startswith("+") else "#f0555e"
            news_html += (
                f"<div class='news-item' style='display: flex; gap: 15px;'>"
                f"<span class='news-date' style='min-width: 80px; flex-shrink: 0; color:{change_color};font-weight:700;'>{change}</span>"
                f"{headline}</div>"
            )
    else:
        news_html += "<div class='news-item' style='color:#8b93a7;'>No movers available.</div>"
    news_html += "</div></div>"
    st.markdown(news_html, unsafe_allow_html=True)


def render_bottom_panels() -> None:
    b1, b2 = st.columns(2)
    checklist_left = get_checklist_left()
    checklist_right = get_checklist_right()

    with b1:
        checklist_html = "<div class='card'><div class='section-title' style='font-size:15px;'>SWING TRADING CHECKLIST</div><div style='margin-top:12px;display:flex;gap:24px;'>"
        checklist_html += "<div>" + "".join([f"<div class='checklist-item'>✅ {item}</div>" for item in checklist_left]) + "</div>"
        checklist_html += "<div>" + "".join([f"<div class='checklist-item'>✅ {item}</div>" for item in checklist_right]) + "</div>"
        checklist_html += "</div></div>"
        st.markdown(checklist_html, unsafe_allow_html=True)

    with b2:
        _draw_market_news(get_market_news())


def render_footer() -> None:
    st.write("")
    updated_at = st.session_state.get("indices_updated_at") or get_indices_updated_at()
    if updated_at is None:
        data_label = "Data from public API"
    else:
        data_label = f"Data as of {updated_at.strftime('%B %d, %Y %I:%M %p PHT')} 🔄"
    st.markdown(
        f"""
        <div style="display:flex;justify-content:space-between;color:#4b5268;font-size:11px;border-top:1px solid #1e2540;padding-top:14px;">
            <span>DISCLAIMER: Information displayed is for reference only and does not constitute investment advice. Please do your own research and consult a licensed financial advisor.</span>
            <span>{data_label}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _draw_live_market() -> None:
    _draw_index_cards(get_indices())
    st.write("")
    st.write("")
    render_watchlist_header()
    render_watchlist_table(get_watchlist_data())
    st.write("")
    st.write("")
    render_bottom_panels()
    render_footer()


@st.fragment(run_every=DASHBOARD_REFRESH)
def render_live_market() -> None:
    """Auto-refresh cards, watchlist, movers, and footer every SCRAPE_INTERVAL."""
    if indices_needs_refresh():
        with st.spinner("Refreshing market data from Binance..."):
            _draw_live_market()
    else:
        _draw_live_market()


def render_dashboard() -> None:
    st.write("")
    render_header()
    st.write("")
    render_live_market()
