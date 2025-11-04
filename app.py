"""
Mining Safety Analysis Platform - Main Streamlit Application
Interactive dashboard for mining accident analysis with AI agents
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime
import os
from pathway_pipeline import initialize_rag_system
from langgraph_agents import create_safety_agent

# Page configuration
st.set_page_config(
    page_title="Mining Safety AI Platform",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .alert-box {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .alert-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
    }
    .alert-warning {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_system():
    """Initialize RAG and Agent systems"""
    with st.spinner("🔄 Initializing AI systems..."):
        rag = initialize_rag_system()
        agent = create_safety_agent(rag)
    return rag, agent


def render_sidebar():
    """Render sidebar with system info"""
    st.sidebar.title("⛏️ Mining Safety AI")
    st.sidebar.markdown("---")
    
    # API Key input
    api_key = st.sidebar.text_input(
        "OpenAI API Key",
        type="password",
        help="Enter your OpenAI API key"
    )
    
    if api_key:
        os.environ["OPENAI_API_KEY"] = api_key
        st.sidebar.success("✅ API Key configured")
    else:
        st.sidebar.warning("⚠️ Please enter API Key")
    
    st.sidebar.markdown("---")
    
    # System statistics
    if 'rag' in st.session_state and st.session_state.rag:
        stats = st.session_state.rag.get_statistics()
        st.sidebar.metric("📄 Documents Loaded", stats["total_documents"])
        st.sidebar.metric("📊 Data Chunks", stats["total_chunks"])
        st.sidebar.metric("💀 Total Casualties", stats.get("total_casualties", 0))
        if stats["years"]:
            st.sidebar.metric("📅 Year Range", f"{min(stats['years'])} - {max(stats['years'])}")
        
        # Show mode (Pathway or Fallback)
        mode = stats.get("mode", "unknown")
        if mode == "pathway_streaming":
            st.sidebar.success("🚀 Pathway Streaming Active")
        elif mode == "fallback":
            st.sidebar.info("📦 Fallback Mode (No Streaming)")
        
        if stats.get("streaming_active"):
            st.sidebar.success("👁️ Real-time Monitoring ON")
        else:
            st.sidebar.info("📡 Static Mode")
    
    st.sidebar.markdown("---")
    st.sidebar.info("""
    **System Architecture:**
    - 🚀 Pathway Streaming RAG
    - 📄 Docling PDF Parser
    - 👁️ Real-time Monitoring
    
    **Multi-Agent System:**
    - 🔍 Inspector Agent
    - ⚖️ Compliance Officer
    - 📊 Safety Analyst
    - 🎓 Training Coordinator
    
    **Features:**
    - 🧠 Pattern Detection
    - 🚨 Automated Alerts
    - 📈 Live Visualizations
    """)


def render_home_tab():
    """Render home/overview tab"""
    st.markdown('<div class="main-header">🛡️ Mining Safety Analysis Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Pathway-Powered Real-time Analysis of DGMS India Mining Accident Records (2016-2022)</div>', unsafe_allow_html=True)
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🚀 Streaming", "Pathway", help="Real-time document processing")
    
    with col2:
        st.metric("🤖 Agents", "4 Specialized", help="Multi-agent collaboration")
    
    with col3:
        st.metric("📄 Parser", "Docling", help="Vision-based PDF parsing")
    
    with col4:
        if 'rag' in st.session_state:
            stats = st.session_state.rag.get_statistics()
            st.metric("📊 Documents", stats["total_documents"])
    
    st.markdown("---")
    
    # Architecture diagram
    st.subheader("🏗️ System Architecture")
    st.code("""
┌─────────────────────────────────────────────────────────────┐
│                    PATHWAY STREAMING LAYER                  │
│  📁 File Monitor → 📄 Docling Parser → 🔪 Chunker →       │
│  🧮 Embeddings → 💾 Vector Index → 🔍 Similarity Search    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                   LANGGRAPH AGENT LAYER                     │
│                                                             │
│  Query → RAG Retrieval → ┌──────────────────────┐         │
│                          │  🔍 Inspector        │         │
│                          │  📊 Safety Analyst   │         │
│                          │  ⚖️ Compliance       │         │
│                          │  🎓 Training Coord   │         │
│                          └──────────────────────┘         │
│                                 ↓                          │
│                    Synthesis & Recommendations            │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│                    STREAMLIT UI LAYER                       │
│  📊 Dashboard | 🔍 Query | 🤖 Agents | 📈 Analytics       │
└─────────────────────────────────────────────────────────────┘
    """, language="")
    
    st.markdown("---")
    
    # Features
    st.subheader("🚀 Platform Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **🔍 Intelligent Analysis**
        - Natural language query interface
        - Semantic search across accident records
        - Pattern detection and trend analysis
        - Root cause identification
        """)
        
        st.markdown("""
        **🤖 Autonomous Agents**
        - Digital Mine Safety Officer
        - Automated incident classification
        - Real-time hazard monitoring
        - Compliance checking
        """)
    
    with col2:
        st.markdown("""
        **📊 Real-time Insights**
        - Live accident trend visualization
        - Geographic hotspot mapping
        - Timeline analysis
        - Statistical dashboards
        """)
        
        st.markdown("""
        **📋 Automated Reporting**
        - Safety audit report generation
        - Actionable recommendations
        - Regulatory compliance reports
        - Preventive measure suggestions
        """)
    
    # Quick start guide
    st.markdown("---")
    st.subheader("🎯 Quick Start Guide")
    
    st.info("""
    1. **Enter your OpenAI API key** in the sidebar
    2. **Place DGMS PDF files** in the `data/dgms/` directory
    3. **Restart the app** to load documents
    4. **Navigate to Query tab** to start analyzing
    5. **Use the Agent tab** for autonomous safety analysis
    """)


def render_query_tab():
    """Render query interface tab"""
    st.header("🔍 Document Query & Search")
    
    if 'rag' not in st.session_state or not st.session_state.rag.documents:
        st.warning("⚠️ No documents loaded. Please place PDF files in `data/dgms/` and restart the app.")
        return
    
    # Query input
    query = st.text_input(
        "Enter your query:",
        placeholder="e.g., Show me all methane-related accidents in 2021 in underground coal mines",
        help="Ask questions about mining accidents, safety patterns, or specific incidents"
    )
    
    col1, col2 = st.columns([1, 4])
    with col1:
        k = st.slider("Results", 1, 10, 4, help="Number of relevant documents to retrieve")
    
    if st.button("🔍 Search", type="primary") and query:
        with st.spinner("Searching documents..."):
            results = st.session_state.rag.query(query, k=k)
        
        if results.get("error"):
            st.error(f"Error: {results['error']}")
        elif results["count"] == 0:
            st.warning("No relevant documents found")
        else:
            st.success(f"Found {results['count']} relevant documents")
            
            # Display results
            for idx, result in enumerate(results["results"], 1):
                with st.expander(f"📄 Result {idx} - {result['source']} ({result['year']})"):
                    st.markdown(result["content"])


def render_agent_tab():
    """Render multi-agent analysis tab"""
    st.header("🤖 Multi-Agent Safety Analysis System")
    st.markdown("*Collaborative AI agents: Inspector | Compliance Officer | Safety Analyst | Training Coordinator*")
    
    if 'agent' not in st.session_state:
        st.error("Agent system not initialized. Please configure API key and restart.")
        return
    
    if 'rag' not in st.session_state or not st.session_state.rag.documents:
        st.warning("⚠️ No documents loaded. Please place PDF files in `data/dgms/` and restart the app.")
        return
    
    # Example queries
    st.markdown("**💡 Example Queries:**")
    example_queries = [
        "Show me all methane-related accidents in 2021",
        "Analyze transportation machinery accidents in Jharkhand",
        "What are the main causes of fatal accidents?",
        "Identify safety compliance gaps in recent reports",
        "Show roof fall incidents and recommend preventive measures"
    ]
    
    cols = st.columns(3)
    for idx, example in enumerate(example_queries[:3]):
        with cols[idx]:
            if st.button(example, key=f"example_{idx}"):
                st.session_state.agent_query = example
    
    st.markdown("---")
    
    # Query input
    query = st.text_area(
        "Enter your safety analysis query:",
        value=st.session_state.get("agent_query", ""),
        placeholder="Ask the Multi-Agent System anything about mining accidents, safety patterns, or compliance...",
        height=100
    )
    
    if st.button("🚀 Analyze with Multi-Agent System", type="primary") and query:
        with st.spinner("🤖 Multi-agent system analyzing... This may take 60-90 seconds..."):
            result = st.session_state.agent.process_query(query)
        
        if result.get("error"):
            st.error(f"Error: {result['error']}")
        else:
            # Display results in structured format
            st.success(f"✅ Analysis Complete - {len(result.get('agents_consulted', []))} agents consulted")
            
            # Show which agents were involved
            if result.get("agents_consulted"):
                st.info(f"🤝 Agents Consulted: {', '.join(result['agents_consulted'])}")
            
            # Alerts section (highest priority)
            if result["alerts"]:
                st.markdown("### 🚨 Priority Alerts")
                for alert in result["alerts"]:
                    if "CRITICAL" in alert or "🚨" in alert:
                        st.markdown(f'<div class="alert-box alert-critical">{alert}</div>', unsafe_allow_html=True)
                    else:
                        st.markdown(f'<div class="alert-box alert-warning">{alert}</div>', unsafe_allow_html=True)
            
            # Create tabs for different agent outputs
            agent_tabs = st.tabs(["🔍 Inspector", "⚖️ Compliance", "📊 Risk Analysis", "🎓 Training", "💡 Recommendations"])
            
            with agent_tabs[0]:
                st.markdown("#### Inspector Agent - Incident Classification")
                if result["classification"]:
                    st.json(result["classification"])
            
            with agent_tabs[1]:
                st.markdown("#### Compliance Officer - Regulatory Check")
                if result["compliance"]:
                    st.json(result["compliance"])
                    
                    violations = result["compliance"].get("violations_detected", [])
                    if violations:
                        st.error(f"⚠️ {len(violations)} regulatory violations detected")
                        for v in violations[:5]:
                            st.markdown(f"- {v}")
            
            with agent_tabs[2]:
                st.markdown("#### Safety Analyst - Pattern & Risk Analysis")
                if result["risk_analysis"]:
                    risk_level = result["risk_analysis"].get("risk_level", "unknown")
                    
                    if risk_level == "critical":
                        st.error(f"🔴 Risk Level: **{risk_level.upper()}**")
                    elif risk_level == "high":
                        st.warning(f"🟠 Risk Level: **{risk_level.upper()}**")
                    else:
                        st.info(f"🟢 Risk Level: **{risk_level.upper()}**")
                    
                    st.json(result["risk_analysis"])
            
            with agent_tabs[3]:
                st.markdown("#### Training Coordinator - Learning & Development")
                if result["training"]:
                    st.json(result["training"])
                    
                    immediate = result["training"].get("immediate_training", [])
                    if immediate:
                        st.warning(f"⏰ {len(immediate)} immediate training needs identified")
            
            with agent_tabs[4]:
                st.markdown("#### Synthesized Recommendations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("**⚡ Immediate Actions**")
                    if result["immediate_actions"]:
                        for action in result["immediate_actions"]:
                            st.markdown(f"- {action}")
                    else:
                        st.info("No immediate actions required")
                
                with col2:
                    st.markdown("**📅 Long-term Recommendations**")
                    if result["long_term_recommendations"]:
                        for rec in result["long_term_recommendations"]:
                            st.markdown(f"- {rec}")
                    else:
                        st.info("No long-term recommendations")
            
            # Source info
            st.markdown("---")
            st.info(f"📚 Analysis based on {result['source_documents']} source documents")


def render_dashboard_tab():
    """Render visualization dashboard with real extracted data"""
    st.header("📊 Safety Analytics Dashboard")
    
    if 'rag' not in st.session_state or not st.session_state.rag.documents:
        st.warning("⚠️ No documents loaded. Please place PDF files in `data/dgms/` and restart the app.")
        return
    
    stats = st.session_state.rag.get_statistics()
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📄 Total Reports", stats["total_documents"])
    with col2:
        st.metric("💀 Total Casualties", stats.get("total_casualties", 0))
    with col3:
        fatal_count = stats["severity_distribution"].get("fatal", 0)
        st.metric("⚠️ Fatal Incidents", fatal_count)
    with col4:
        if stats["years"]:
            year_span = int(max(stats["years"])) - int(min(stats["years"])) + 1
            st.metric("📅 Years Covered", year_span)
    
    st.markdown("---")
    
    # Filter section
    st.subheader("🔍 Filter Data")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        year_options = ["All"] + stats["years"]
        selected_year = st.selectbox("Year", year_options)
    
    with col2:
        state_options = ["All"] + list(stats["states"].keys())
        selected_state = st.selectbox("State", state_options)
    
    with col3:
        severity_options = ["All", "fatal", "serious", "minor"]
        selected_severity = st.selectbox("Severity", severity_options)
    
    # Apply filters
    filter_year = None if selected_year == "All" else selected_year
    filter_state = None if selected_state == "All" else selected_state
    filter_severity = None if selected_severity == "All" else selected_severity
    
    filtered_data = st.session_state.rag.get_filtered_data(
        year=filter_year,
        state=filter_state,
        severity=filter_severity
    )
    
    st.info(f"Showing {len(filtered_data)} incidents matching filters")
    
    st.markdown("---")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Yearly trend
        if stats["years"] and len(stats["years"]) > 1:
            year_counts = {}
            for doc_meta in filtered_data:
                year = doc_meta.get("year", "unknown")
                if year != "unknown":
                    year_counts[year] = year_counts.get(year, 0) + 1
            
            if year_counts:
                df_years = pd.DataFrame(list(year_counts.items()), columns=["Year", "Incidents"])
                df_years = df_years.sort_values("Year")
                
                fig = px.line(df_years, x="Year", y="Incidents",
                             title="📈 Incident Trend Over Time",
                             markers=True)
                fig.update_traces(line_color='#1f77b4', line_width=3)
                st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Severity distribution
        severity_data = stats["severity_distribution"]
        if severity_data:
            df_severity = pd.DataFrame(list(severity_data.items()), 
                                       columns=["Severity", "Count"])
            
            colors = {'fatal': '#d62728', 'serious': '#ff7f0e', 'minor': '#2ca02c'}
            df_severity['Color'] = df_severity['Severity'].map(colors)
            
            fig = px.bar(df_severity, x="Severity", y="Count",
                        title="⚠️ Severity Distribution",
                        color="Severity",
                        color_discrete_map=colors)
            st.plotly_chart(fig, use_container_width=True)
    
    # Second row
    col1, col2 = st.columns(2)
    
    with col1:
        # Incident types
        if stats["incident_types"]:
            # Get top 7 incident types
            incident_items = sorted(stats["incident_types"].items(), 
                                   key=lambda x: x[1], reverse=True)[:7]
            df_types = pd.DataFrame(incident_items, columns=["Type", "Count"])
            
            fig = px.pie(df_types, values="Count", names="Type",
                        title="🔧 Incident Type Distribution",
                        hole=0.3)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Geographic distribution
        if stats["states"]:
            # Get top 6 states
            state_items = sorted(stats["states"].items(), 
                                key=lambda x: x[1], reverse=True)[:6]
            df_states = pd.DataFrame(state_items, columns=["State", "Incidents"])
            
            fig = px.bar(df_states, x="Incidents", y="State",
                        title="🗺️ Geographic Hotspots",
                        orientation='h',
                        color="Incidents",
                        color_continuous_scale="Reds")
            st.plotly_chart(fig, use_container_width=True)
    
    # Detailed data table
    st.markdown("---")
    st.subheader("📋 Detailed Incident Records")
    
    if filtered_data:
        # Prepare table data
        table_data = []
        for meta in filtered_data[:50]:  # Limit to 50 rows
            table_data.append({
                "Source": meta.get("source", "N/A"),
                "Year": meta.get("year", "N/A"),
                "State": meta.get("state", "N/A"),
                "Severity": meta.get("severity", "N/A"),
                "Mine Type": meta.get("mine_type", "N/A"),
                "Casualties": meta.get("casualties", 0),
                "Incident Types": ", ".join(meta.get("incident_types", ["N/A"])[:2])
            })
        
        df_table = pd.DataFrame(table_data)
        st.dataframe(df_table, use_container_width=True, height=400)
        
        # Download button
        csv = df_table.to_csv(index=False)
        st.download_button(
            label="📥 Download Data as CSV",
            data=csv,
            file_name="mining_incidents_filtered.csv",
            mime="text/csv"
        )
    else:
        st.info("No data available for the selected filters")


def main():
    """Main application"""
    
    # Initialize session state
    if 'initialized' not in st.session_state:
        try:
            rag, agent = initialize_system()
            st.session_state.rag = rag
            st.session_state.agent = agent
            st.session_state.initialized = True
        except Exception as e:
            st.error(f"Initialization error: {e}")
            st.info("Please ensure OpenAI API key is set and documents are in data/dgms/")
            return
    
    # Render sidebar
    render_sidebar()
    
    # Main content tabs
    tabs = st.tabs(["🏠 Home", "🔍 Query", "🤖 AI Agent", "📊 Dashboard"])
    
    with tabs[0]:
        render_home_tab()
    
    with tabs[1]:
        render_query_tab()
    
    with tabs[2]:
        render_agent_tab()
    
    with tabs[3]:
        render_dashboard_tab()


if __name__ == "__main__":
    main()