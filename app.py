"""
SARSA - AI for Mine Safety Intelligence (Streamlit)
- Uses OpenAI (LangChain) for conversational answers (set OPENAI_API_KEY).
- Live mode auto-refresh + token streaming.
- Reads live JSONL rows produced by the Pathway ingestor and shows a 🚨 Live Events column.

Run:
    export OPENAI_API_KEY=sk-...
    streamlit run app.py
"""

import os
from dotenv import load_dotenv
load_dotenv()

import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import time
from datetime import datetime

from data_processor import PDFAccidentParser, PathwayStreamSimulator, RegulatoryKnowledgeBase
from agents import SARSAAgentOrchestrator

# ---- Live JSONL loader (Pathway output) ----
import glob

def load_streamed_rows(jsonl_dir: str = "out/jsonl") -> pd.DataFrame:
    paths = sorted(glob.glob(f"{jsonl_dir}/**/*.jsonl", recursive=True)) + sorted(glob.glob(f"{jsonl_dir}/*.jsonl"))
    frames = []
    for p in paths:
        try:
            frames.append(pd.read_json(p, lines=True))
        except Exception:
            pass
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    # Standardize types
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], errors='coerce')
        df['year'] = df['date'].dt.year
    # Dedup on 'id' if present
    if 'id' in df.columns:
        df = df.drop_duplicates(subset=['id'], keep='last')
    return df

def merge_new_data(base_df: pd.DataFrame, streamed_df: pd.DataFrame):
    if base_df is None or len(base_df) == 0:
        return streamed_df.copy(), streamed_df.copy()
    if streamed_df is None or len(streamed_df) == 0:
        return base_df, pd.DataFrame()
    if 'id' in base_df.columns and 'id' in streamed_df.columns:
        existing_ids = set(base_df['id'].tolist())
        new_rows = streamed_df[~streamed_df['id'].isin(existing_ids)].copy()
        merged = pd.concat([base_df, new_rows], ignore_index=True)
        return merged, new_rows
    merged = pd.concat([base_df, streamed_df], ignore_index=True).drop_duplicates()
    return merged, streamed_df

# ---------------- UI setup ----------------
st.set_page_config(layout="wide", page_title="SARSA - Mine Safety Intelligence Platform")

st.markdown("""
<style>
    .main-header { font-size: 2.5rem; font-weight: bold; color: #1f77b4; margin-bottom: 0.5rem; }
    .sub-header { font-size: 1.2rem; color: #666; margin-bottom: 1.0rem; }
</style>
""", unsafe_allow_html=True)

# Session
if 'file_processed' not in st.session_state:
    st.session_state['file_processed'] = False
if 'df' not in st.session_state:
    st.session_state['df'] = None
if 'orchestrator' not in st.session_state:
    st.session_state['orchestrator'] = None
if 'pathway_stream' not in st.session_state:
    st.session_state['pathway_stream'] = None
if 'regulatory_kb' not in st.session_state:
    st.session_state['regulatory_kb'] = RegulatoryKnowledgeBase()

# Title
st.markdown('<div class="main-header">⛏️ SARSA - AI for Mine Safety Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Agentic analytics + semantic RAG + OpenAI conversational answers</div>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🤖 SARSA Agent Control")

    st.subheader("📄 Data Ingestion (Manual)")
    uploaded_file = st.file_uploader("Upload DGMS Report (PDF)", type=["pdf"])

    if uploaded_file and not st.session_state['file_processed']:
        if st.button("🚀 Process PDF & Initialize Agents"):
            with st.spinner("Processing PDF → Building embeddings → Initializing agents..."):
                parser = PDFAccidentParser()
                df = parser.parse_pdf(uploaded_file)
                if len(df) == 0:
                    st.error("⚠️ No accident data found in PDF.")
                else:
                    stream = PathwayStreamSimulator()
                    stream.ingest_data(df)
                    orch = SARSAAgentOrchestrator(
                        data=df,
                        pathway_stream=stream,
                        regulatory_kb=st.session_state['regulatory_kb'],
                    )
                    st.session_state['df'] = df
                    st.session_state['pathway_stream'] = stream
                    st.session_state['orchestrator'] = orch
                    st.session_state['file_processed'] = True
                    st.success(f"✅ Processed {len(df)} accident records!")
                    time.sleep(0.7)
                    st.rerun()

    st.subheader("📡 Live Streaming (Pathway FS)")
    st.caption("Run `python pathway_ingestor_fs.py` in another terminal to stream `data/dgms/` → `out/jsonl/`")

    st.subheader("🔑 LLM Status")
    if os.getenv("OPENAI_API_KEY"):
        st.success("OpenAI: Connected (OPENAI_API_KEY detected)")
    else:
        st.info("OpenAI: Missing OPENAI_API_KEY")

    st.subheader("🔄 Live")
    live = st.checkbox("Enable live mode", value=False, help="Auto-refresh the page to poll for new data/alerts")
    interval = st.number_input("Refresh interval (sec)", min_value=2, max_value=60, value=8, step=1, help="How often to refresh when live")

    st.subheader("🧭 Navigation")
    if st.session_state['file_processed']:
        st.session_state.page = st.radio("Go to", ("📊 Executive Dashboard", "💬 Interactive Query", "📝 Generate Safety Audit"), key="navigation")
    else:
        st.session_state.page = None
        st.info("👆 Upload a DGMS report to activate analysis tools")

# Main
if not st.session_state['file_processed']:
    st.info("### 🚀 Welcome to SARSA - Mine Safety Intelligence Platform")
    st.write("Upload a DGMS report to begin, or run the Pathway FS ingestor to stream files live.")
else:
    df = st.session_state['df']
    orchestrator = st.session_state['orchestrator']

    # Auto-refresh when live mode is enabled
    if live:
        # Pull streamed rows and merge before rendering pages
        streamed = load_streamed_rows("out/jsonl")
        merged_df, new_rows = merge_new_data(df, streamed)
        if len(new_rows) > 0:
            st.session_state['df'] = merged_df
            df = merged_df
            # reindex embeddings
            if st.session_state.get('pathway_stream'):
                st.session_state['pathway_stream'].ingest_data(merged_df)
            for _, r in new_rows.sort_values('date', ascending=False).head(5).iterrows():
                msg = f"{r.get('date','')} — {r.get('cause_category','Unknown')} at {r.get('mine_name','Unknown')} ({r.get('state','')})"
                st.toast(msg, icon="⚠️")
        time.sleep(int(interval))
        st.rerun()

    # Page 1: Dashboard
    if st.session_state.page == "📊 Executive Dashboard":
        st.subheader("📊 Executive Dashboard")
        with st.spinner("🤖 Analysis Agent processing..."):
            analysis = orchestrator.run_analysis()

        kpis = analysis['kpis']
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Fatal Accidents", kpis.get('total_accidents', 0))
        col2.metric("Total Deaths", kpis.get('total_deaths', 0))
        col3.metric("Primary Hazard", kpis.get('primary_hazard', '—'))
        col4.metric("Highest-Risk State", kpis.get('highest_risk_state', '—'))

        st.divider()

        col_left, col_right = st.columns([3, 1])
        with col_left:
            st.markdown("#### 🗺️ Accident Location Heatmap")
            if 'lat' in df.columns and 'lon' in df.columns and df[['lat','lon']].notna().any().any():
                map_df = df.dropna(subset=['lat','lon']).copy()
                view_state = pdk.ViewState(latitude=map_df['lat'].mean(), longitude=map_df['lon'].mean(), zoom=4, pitch=40)
                layer = pdk.Layer('ScatterplotLayer', data=map_df, get_position='[lon, lat]', get_color='[200, 30, 0, 160]', get_radius='deaths * 15000', pickable=True)
                tooltip = {"html": "<b>{state}</b><br/>Mine: {mine_name}<br/>Deaths: {deaths}", "style": {"backgroundColor": "steelblue", "color": "white"}}
                st.pydeck_chart(pdk.Deck(map_style=None, initial_view_state=view_state, layers=[layer], tooltip=tooltip))
            else:
                st.info("Geographic data not available for mapping")

            if 'date' in df.columns:
                st.markdown("#### 📅 Accident Timeline")
                timeline_chart = alt.Chart(df).mark_circle(size=60).encode(
                    x=alt.X('date:T', title='Date'),
                    y=alt.Y('cause_category:N', title='Cause Category'),
                    color=alt.Color('state:N', title='State'),
                    tooltip=['date', 'mine_name', 'cause_category', 'state']
                ).properties(height=300).interactive()
                st.altair_chart(timeline_chart, use_container_width=True)

        with col_right:
            st.markdown("#### 🚨 Live Events")
            streamed = load_streamed_rows("out/jsonl")
            merged_df, new_rows = merge_new_data(df, streamed)
            if len(new_rows) > 0:
                st.success(f"New incidents: {len(new_rows)}")
                # Update session & reindex
                st.session_state['df'] = merged_df
                if st.session_state.get('pathway_stream'):
                    st.session_state['pathway_stream'].ingest_data(merged_df)
                show_cols = ['date','mine_name','state','cause_category','deaths']
                if all(c in new_rows.columns for c in show_cols):
                    show = new_rows[show_cols].copy()
                else:
                    show = new_rows.head(10)
                st.dataframe(show.sort_values('date', ascending=False).head(8), use_container_width=True, height=360)
                for _, r in new_rows.sort_values('date', ascending=False).head(5).iterrows():
                    msg = f"{r.get('date','')} — {r.get('cause_category','Unknown')} at {r.get('mine_name','Unknown')} ({r.get('state','')})"
                    st.toast(msg, icon="⚠️")
            else:
                st.caption("No new incidents yet.")

    # Page 2: Interactive Query
    elif st.session_state.page == "💬 Interactive Query":
        st.subheader("💬 Agentic Query Interface")
        query = st.text_input("🔍 Enter your query:", key="query_input", placeholder="e.g., summarize accidents in 2020")
        if query:
            with st.spinner("🤖 Query Agent processing..."):
                result = orchestrator.run_query(query)
            st.markdown("### 🗣️ Natural Language Answer (OpenAI, live tokens)")
            try:
                st.write_stream(orchestrator.to_natural_language_stream(query, result))
            except Exception:
                nl = orchestrator.to_natural_language(query, result)
                st.write(nl)
            st.markdown("### 📋 Structured Result (reference)")
            st.json(result, expanded=False)

    # Page 3: Safety Audit
    elif st.session_state.page == "📝 Generate Safety Audit":
        st.subheader("📝 Automated Safety Audit Generation")
        if st.button("🚀 Generate Comprehensive Safety Audit Report"):
            with st.spinner("🤖 Audit Agent analyzing..."):
                audit = orchestrator.run_audit()
            st.markdown("### 🗣️ Executive Summary")
            try:
                st.write_stream(orchestrator.to_natural_language_stream("Provide an executive summary of this audit report", audit))
            except Exception:
                summary_text = orchestrator.to_natural_language("Provide an executive summary of this audit report", audit)
                st.write(summary_text)
            st.markdown("---")
            st.markdown("### 📦 Full Structured Audit (JSON)")
            st.json(audit, expanded=False)

st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
  <p><strong>SARSA</strong> - AI for Mine Safety Intelligence | Powered by LangGraph + OpenAI | Live stream via Pathway FS</p>
</div>
""", unsafe_allow_html=True)
