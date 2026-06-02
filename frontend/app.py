"""TrialBridge — Frontend (Streamlit)
Full implementation coming in Week 4.
"""
import streamlit as st

st.set_page_config(
    page_title="TrialBridge",
    page_icon="🧬",
    layout="wide",
)

st.title("🧬 TrialBridge")
st.subheader("AI-Powered Clinical Trial Matching")
st.info("Infrastructure is live. NLP pipeline and matching engine coming in Weeks 2–3.")

col1, col2, col3 = st.columns(3)
col1.metric("Trials Indexed", "0", help="Will populate after Day 3 ingestion pipeline")
col2.metric("Matches Made", "0")
col3.metric("API Status", "🟢 Online")
