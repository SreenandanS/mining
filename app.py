"""
SARSA - AI for Mine Safety Intelligence
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import altair as alt
import pydeck as pdk
import time
import re
from datetime import datetime

# Import our custom modules
from data_processor import PDFAccidentParser, PathwayStreamSimulator, RegulatoryKnowledgeBase
from agents import SARSAAgentOrchestrator

# Set page config
st.set_page_config(layout="wide", page_title="SARSA - Mine Safety Intelligence Platform")

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .alert-box {
        padding: 0.75rem;
        border-radius: 0.25rem;
        margin-bottom: 0.5rem;
    }
    .alert-warning {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
    }
    .alert-info {
        background-color: #d1ecf1;
        border-left: 4px solid #17a2b8;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
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

# --- Page Title ---
st.markdown('<div class="main-header">⛏️ SARSA - AI for Mine Safety Intelligence</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Leveraging Agentic AI and Real-Time RAG for Safer Mining Operations</div>', unsafe_allow_html=True)

# --- Sidebar ---
with st.sidebar:
    st.header("🤖 SARSA Agent Control")
    
    # 1. File Uploader
    st.subheader("📄 Data Ingestion")
    uploaded_file = st.file_uploader(
        "Upload DGMS Report (PDF)", 
        type=["pdf"],
        help="Upload DGMS accident report PDF for analysis"
    )
    
    if uploaded_file and not st.session_state['file_processed']:
        if st.button("🚀 Process PDF & Initialize Agents"):
            with st.spinner("Processing PDF... Extracting entities... Building knowledge graph... Initializing agents..."):
                try:
                    # Parse PDF
                    parser = PDFAccidentParser()
                    df = parser.parse_pdf(uploaded_file)
                    
                    if len(df) == 0:
                        st.error("⚠️ No accident data found in PDF. Please check the file format.")
                    else:
                        # Initialize Pathway stream simulator
                        pathway_stream = PathwayStreamSimulator()
                        pathway_stream.ingest_data(df)
                        
                        # Initialize agents
                        orchestrator = SARSAAgentOrchestrator(
                            data=df,
                            pathway_stream=pathway_stream,
                            regulatory_kb=st.session_state['regulatory_kb']
                        )
                        
                        # Store in session state
                        st.session_state['df'] = df
                        st.session_state['pathway_stream'] = pathway_stream
                        st.session_state['orchestrator'] = orchestrator
                        st.session_state['file_processed'] = True
                        st.session_state['total_accidents'] = len(df)
                        st.session_state['total_deaths'] = df['deaths'].sum() if 'deaths' in df.columns else len(df)
                        
                        st.success(f"✅ Processed {len(df)} accident records!")
                        time.sleep(1)
                        st.rerun()
                        
                except Exception as e:
                    st.error(f"❌ Error processing PDF: {str(e)}")
    
    # 2. System Status
    st.subheader("📊 System Status")
    with st.expander("View Live System Status", expanded=True):
        if st.session_state['file_processed']:
            status_color = "green"
            status_text = "✅ Online"
        else:
            status_color = "orange"
            status_text = "⏳ Awaiting Data"
        
        st.markdown(f"**Pathway Ingestion:** <span style='color:{status_color};'>{status_text}</span>", unsafe_allow_html=True)
        st.markdown(f"**LangGraph Agents:** <span style='color:{status_color};'>{status_text}</span>", unsafe_allow_html=True)
        st.markdown(f"**Vector DB (FAISS):** <span style='color:{status_color};'>{status_text}</span>", unsafe_allow_html=True)
        st.markdown(f"**Knowledge Graph:** <span style='color:{status_color};'>{status_text}</span>", unsafe_allow_html=True)
        
        if st.session_state['file_processed']:
            st.markdown(f"**Records Indexed:** <span style='color:green;'>{st.session_state['total_accidents']}</span>", unsafe_allow_html=True)
    
    # 3. Live Alert Feed
    st.subheader("🔔 Live Alert Feed")
    with st.expander("View Real-Time Alerts", expanded=False):
        if st.session_state['file_processed'] and st.session_state['pathway_stream']:
            alerts = st.session_state['pathway_stream'].generate_live_alerts()
            
            if len(alerts) > 0:
                for alert in alerts[:5]:  # Show top 5
                    if alert['type'] == 'warning':
                        st.warning(f"**ALERT ({alert['time']}):** {alert['message']}")
                    else:
                        st.info(f"**INFO ({alert['time']}):** {alert['message']}")
            else:
                st.info("No recent alerts")
        else:
            st.info("Upload PDF to activate live monitoring")
    
    # 4. Navigation
    st.divider()
    if st.session_state['file_processed']:
        st.session_state.page = st.radio(
            "🧭 Navigation",
            ("📊 Executive Dashboard", "💬 Interactive Query", "📝 Generate Safety Audit"),
            key="navigation"
        )
    else:
        st.session_state.page = None
        st.info("👆 Upload a DGMS report to activate analysis tools")

# --- Main Content Area ---
if not st.session_state['file_processed']:
    # Landing Page
    st.info("### 🚀 Welcome to SARSA - Mine Safety Intelligence Platform")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🤖 Analysis Agent")
        st.write("Generates live KPIs, heatmaps, and trend analysis")
        st.write("✓ Real-time data processing")
        st.write("✓ Geographic risk mapping")
        st.write("✓ Temporal trend analysis")
    
    with col2:
        st.markdown("#### 💬 Query Agent")
        st.write("Natural language interface for safety inquiries")
        st.write("✓ Semantic search using RAG")
        st.write("✓ Entity recognition")
        st.write("✓ Complex query interpretation")
    
    with col3:
        st.markdown("#### 📝 Audit Agent")
        st.write("Automated safety compliance reports")
        st.write("✓ DGMS regulation cross-reference")
        st.write("✓ Root cause analysis")
        st.write("✓ Actionable recommendations")
    
    st.divider()
    st.markdown("### 📋 How It Works")
    st.write("""
    1. **Upload** your DGMS accident report (PDF format)
    2. **AI Pipeline** extracts and structures accident data using PyMuPDF + NLP
    3. **Pathway** creates real-time data streams and builds knowledge graph
    4. **LangGraph Agents** provide intelligent analysis, queries, and audits
    5. **Explore** insights through interactive dashboard and natural language queries
    """)
    
    st.info("👈 Get started by uploading a DGMS PDF report in the sidebar")

else:
    df = st.session_state['df']
    orchestrator = st.session_state['orchestrator']
    
    # --- Page 1: Executive Dashboard ---
    if st.session_state.page == "📊 Executive Dashboard":
        st.subheader("📊 Executive Dashboard")
        
        # Run Analysis Agent
        with st.spinner("🤖 Analysis Agent processing..."):
            analysis = orchestrator.run_analysis()
        
        kpis = analysis['kpis']
        
        # KPI Cards
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Fatal Accidents", kpis['total_accidents'])
        col2.metric("Total Deaths", kpis['total_deaths'])
        col3.metric("Primary Hazard", kpis['primary_hazard'])
        col4.metric("Highest-Risk State", kpis['highest_risk_state'])
        
        st.divider()
        
        # Charts
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown("#### 📈 Accidents by Cause Category")
            if 'cause_category' in df.columns:
                cause_chart = alt.Chart(df).mark_bar().encode(
                    x=alt.X('count()', title='Number of Accidents'),
                    y=alt.Y('cause_category:N', title='Cause Category', sort='-x'),
                    color=alt.Color('cause_category:N', legend=None),
                    tooltip=['cause_category', 'count()']
                ).properties(height=300).interactive()
                st.altair_chart(cause_chart, use_container_width=True)
            else:
                st.info("No cause category data available")
        
        with col2:
            st.markdown("#### 🗺️ Accidents by State")
            if 'state' in df.columns:
                state_chart = alt.Chart(df).mark_bar(color='#FF8C00').encode(
                    x=alt.X('count()', title='Number of Accidents'),
                    y=alt.Y('state:N', title='State', sort='-x'),
                    tooltip=['state', 'count()']
                ).properties(height=300).interactive()
                st.altair_chart(state_chart, use_container_width=True)
            else:
                st.info("No state data available")
        
        st.divider()
        
        # Geographic Map
        st.markdown("#### 🗺️ Accident Location Heatmap")
        if len(analysis['heatmaps']) > 0:
            map_data = pd.DataFrame(analysis['heatmaps'])
            
            view_state = pdk.ViewState(
                latitude=map_data['lat'].mean(),
                longitude=map_data['lon'].mean(),
                zoom=4,
                pitch=40,
            )
            
            layer = pdk.Layer(
                'ScatterplotLayer',
                data=map_data,
                get_position='[lon, lat]',
                get_color='[200, 30, 0, 160]',
                get_radius='deaths * 15000',
                pickable=True
            )
            
            tooltip = {
                "html": "<b>{state}</b><br/>Accidents: {accidents}<br/>Deaths: {deaths}",
                "style": {"backgroundColor": "steelblue", "color": "white"}
            }
            
            st.pydeck_chart(pdk.Deck(
                map_style=None,
                initial_view_state=view_state,
                layers=[layer],
                tooltip=tooltip
            ))
        else:
            st.info("Geographic data not available for mapping")
        
        # Timeline
        if 'date' in df.columns:
            st.markdown("#### 📅 Accident Timeline")
            timeline_chart = alt.Chart(df).mark_circle(size=60).encode(
                x=alt.X('date:T', title='Date'),
                y=alt.Y('cause_category:N', title='Cause Category'),
                color=alt.Color('state:N', title='State'),
                tooltip=['date', 'mine_name', 'cause_category', 'state']
            ).properties(height=300).interactive()
            st.altair_chart(timeline_chart, use_container_width=True)
    
    # --- Page 2: Interactive Query ---
    elif st.session_state.page == "💬 Interactive Query":
        st.subheader("💬 Agentic Query Interface")
        st.write("Ask the Query Agent about safety incidents. Try queries like:")
        st.code("• 'Give me a summary'\n• 'Tell me about accidents in Jharkhand'\n• 'Show dumper-related incidents'\n• 'What happened at Chikla Mine'")
        
        query = st.text_input("🔍 Enter your query:", key="query_input", placeholder="e.g., summarize accidents in 2020")
        
        if query:
            with st.spinner("🤖 Query Agent processing..."):
                result = orchestrator.run_query(query)
            
            st.markdown(f"### 🤖 Agent Response")
            
            if result['type'] == 'summary':
                st.info(result['message'])
                
                if result['data']:
                    data = result['data']
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**📊 Total Statistics**")
                        st.write(f"- Fatal Accidents: `{data['total_accidents']}`")
                        st.write(f"- Total Deaths: `{data['total_deaths']}`")
                    
                    with col2:
                        st.markdown("**🔴 Top Hazards**")
                        for cause in data.get('top_causes', [])[:3]:
                            st.write(f"- {cause['cause']}: `{cause['count']}` incidents")
                    
                    if data.get('top_states'):
                        st.markdown("**🗺️ High-Risk States**")
                        for state_info in data['top_states'][:3]:
                            st.write(f"- {state_info['state']}: `{state_info['count']}` incidents")
            
            elif result['type'] == 'entity':
                st.info(result['message'])
                
                if result['data']:
                    for incident in result['data']:
                        with st.expander(f"📍 {incident['mine_name']} - {incident['date']}"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.write(f"**Location:** {incident['location']}")
                                st.write(f"**Owner:** {incident['owner']}")
                            with col2:
                                st.write(f"**Hazard:** {incident['cause']}")
                                st.write(f"**Victim:** {incident['victim']}")
                            st.write(f"**Description:** {incident['description']}")
            
            elif result['type'] == 'keyword':
                if result.get('analysis'):
                    analysis = result['analysis']
                    st.info(f"🔍 Found **{analysis['total_matches']}** matching incidents. Primary hazard: **{analysis['primary_hazard']}**, High concentration in: **{analysis['high_risk_state']}")
                else:
                    st.info(result['message'])
                
                if result['data']:
                    st.markdown("### 📋 Matching Incidents")
                    for incident in result['data'][:10]:
                        with st.expander(f"📅 {incident['date']} - {incident['mine_name']} ({incident['cause']})"):
                            st.write(f"**Location:** {incident['location']}")
                            st.write(f"**Description:** {incident['description']}")
            
            else:
                st.info(result['message'])
                if result.get('data'):
                    for item in result['data']:
                        with st.expander(f"{item.get('date', 'Unknown')} - {item.get('mine_name', 'Unknown')}"):
                            st.write(item.get('description', 'No description'))
    
    # --- Page 3: Generate Safety Audit ---
    elif st.session_state.page == "📝 Generate Safety Audit":
        st.subheader("📝 Automated Safety Audit Generation")
        st.write("The Audit Agent analyzes the dataset to identify trends, root causes, and recommend preventive actions aligned with DGMS regulations.")
        
        if st.button("🚀 Generate Comprehensive Safety Audit Report"):
            with st.spinner("🤖 Audit Agent analyzing... Cross-referencing regulations... Generating report..."):
                audit = orchestrator.run_audit()
                time.sleep(2)  # Simulate processing
            
            # Display Audit Report
            st.markdown("---")
            st.markdown(f"## 📋 DGMS Automated Safety Audit Report")
            st.markdown(f"**Report ID:** `{audit['report_id']}`")
            st.markdown(f"**Generated:** `{datetime.fromisoformat(audit['generated_at']).strftime('%Y-%m-%d %H:%M:%S')}`")
            st.markdown("---")
            
            # Executive Summary
            st.markdown("### 1️⃣ Executive Summary")
            summary = audit['executive_summary']
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Total Accidents", summary['total_accidents'])
            col2.metric("Total Deaths", summary['total_deaths'])
            col3.metric("Date Range", summary.get('date_range', 'N/A'))
            
            if summary.get('top_hazards'):
                st.markdown("**🔴 Primary Hazards:**")
                for hazard, count in list(summary['top_hazards'].items())[:3]:
                    st.write(f"- {hazard}: `{count}` incidents")
            
            st.divider()
            
            # Hazard Analysis
            st.markdown("### 2️⃣ Key Hazard Trends & Root Cause Analysis")
            
            if audit['hazard_analysis']:
                for hazard in audit['hazard_analysis']:
                    severity_color = "🔴" if hazard['severity'] == 'Critical' else "🟠" if hazard['severity'] == 'High' else "🟡"
                    
                    with st.expander(f"{severity_color} {hazard['hazard']} - {hazard['deaths']} deaths ({hazard['incidents']} incidents)"):
                        st.markdown(f"**Severity:** `{hazard['severity']}`")
                        st.markdown(f"**Regulation Violated:** `{hazard['regulation_violated']}`")
                        st.markdown(f"**Root Cause:** {hazard['root_cause']}")
            else:
                st.info("No hazard analysis available")
            
            st.divider()
            
            # Compliance Check
            st.markdown("### 3️⃣ Regulatory Compliance Assessment")
            
            if audit['compliance_check']:
                compliance_df = pd.DataFrame(audit['compliance_check'])
                
                st.dataframe(
                    compliance_df[['regulation', 'title', 'violation_count', 'hazard_type', 'status', 'priority']],
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No compliance issues detected")
            
            st.divider()
            
            # Recommendations
            st.markdown("### 4️⃣ Agent-Recommended Preventive Actions")
            
            if audit['recommendations']:
                for i, rec in enumerate(audit['recommendations'], 1):
                    priority_icon = "🚨" if rec['priority'] == 'Critical' else "⚠️"
                    st.markdown(f"**{i}. {priority_icon} {rec['action']}**")
                    st.write(f"   - Compliance: `{rec['regulation']}`")
                    st.write(f"   - Priority: `{rec['priority']}`")
                    st.write("")
            else:
                st.info("No specific recommendations generated")
            
            st.markdown("---")
            st.markdown("*End of Report - Generated by SARSA Audit Agent*")
            
            # Download button
            report_text = f"""
DGMS Automated Safety Audit Report
Report ID: {audit['report_id']}
Generated: {audit['generated_at']}

=== EXECUTIVE SUMMARY ===
Total Accidents: {summary['total_accidents']}
Total Deaths: {summary['total_deaths']}
Date Range: {summary.get('date_range', 'N/A')}

=== HAZARD ANALYSIS ===
{chr(10).join([f"- {h['hazard']}: {h['deaths']} deaths, Root Cause: {h['root_cause']}" for h in audit['hazard_analysis']])}

=== RECOMMENDATIONS ===
{chr(10).join([f"{i}. {rec['action']} (Priority: {rec['priority']})" for i, rec in enumerate(audit['recommendations'], 1)])}
"""
            
            st.download_button(
                label="📥 Download Report as TXT",
                data=report_text,
                file_name=f"SARSA_Audit_{audit['report_id']}.txt",
                mime="text/plain"
            )

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: #666; font-size: 0.9rem;'>
    <p><strong>SARSA</strong> - AI for Mine Safety Intelligence | Built with Pathway, LangGraph, and Streamlit</p>
    <p>Indian Institute of Technology (ISM) Dhanbad</p>
</div>
""", unsafe_allow_html=True)
