"""LASA Guard: a look-alike / sound-alike medicine name checker for Indian brands (multi-page dashboard)."""
import os
import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)                                   # model files are found relative to the project folder
sys.path.insert(0, str(ROOT / "lassa"))
sys.path.insert(0, str(ROOT / "app"))
import ui       # noqa: E402
import views    # noqa: E402

st.set_page_config(page_title="LASA Guard", page_icon="💊", layout="wide", initial_sidebar_state="expanded")

if "dark" not in st.session_state:
    try:
        st.session_state["dark"] = st.context.theme.type == "dark"
    except Exception:
        st.session_state["dark"] = False

PAGES = {
    "🏠 Overview": views.overview,
    "🔎 Check a Medicine": views.check_page,
    "🆕 New Brand Name Check": views.newname_page,
    "🧬 PCA: the core": views.pca_page,
    "📊 Validation": views.validation_page,
    "⚠️ Riskiest Pairs": views.pairs_page,
    "ℹ️ Limitations & About": views.about_page,
}

with st.sidebar:
    st.markdown("### 📌 Navigation")
    page = st.radio("Page", list(PAGES), label_visibility="collapsed")
    st.toggle("🌙 Dark mode", key="dark")
    st.markdown("---")
    st.caption("**LASA Guard**  \nFoundations of ML · NMIMS MPSTME  \nPCA, kNN and k-means written in NumPy.  \nData: 11,825 Indian medicine products; ISMP confused-names list for validation.")

st.markdown(ui.css(st.session_state["dark"]), unsafe_allow_html=True)
PAGES[page](st.session_state["dark"])
