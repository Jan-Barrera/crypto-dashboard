from pathlib import Path

import streamlit as st


def apply_global_styles() -> None:
    css_path = Path(__file__).resolve().parent / "style.css"
    css_content = css_path.read_text(encoding="utf-8")
    st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)