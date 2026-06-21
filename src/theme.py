from __future__ import annotations

from pathlib import Path
import streamlit as st


def load_theme() -> None:
    css_path = Path("assets/styles.css")
    if css_path.exists():
        css = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def section_header(title: str, subtitle: str = "") -> None:
    st.markdown(f"## {title}")
    if subtitle:
        st.markdown(f"<div class='soft-note'>{subtitle}</div>", unsafe_allow_html=True)


def info_card(title: str, body: str) -> None:
    st.markdown(
        f"""
        <div class="neon-card">
            <h4 style="margin-top:0">{title}</h4>
            <div class="soft-note">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )