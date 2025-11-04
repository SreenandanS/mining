"""
SARSA Agentic AI - LangGraph Implementation
Three intelligent agents: Analysis Agent, Query Agent, Audit Agent
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
import pandas as pd
from datetime import datetime
import json
import operator

# LangGraph imports
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    LANGGRAPH_AVAILABLE = True
except:
    LANGGRAPH_AVAILABLE = False
    print("LangGraph not available - using fallback implementation")


class AgentState(TypedDict):
    """State shared across all agents"""
    messages: Annotated[List, operator.add]
    data: pd.DataFrame
    query: str
    result: Dict
    agent_history: List[str]


class AnalysisAgent:
    """
    Agent 1: Analysis Agent
    - Generates live KPIs and trends
    - Creates heatmaps of accident patterns  
    - Real-time data processing
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.name = "Analysis Agent"
    
    def analyze(self) -> Dict:
        """Generate comprehensive analysis"""
        
        if len(self.data) == 0:
            return self._empty_analysis()
        
        analysis = {
            'kpis': self._generate_kpis(),
            'trends': self._analyze_trends(),
            'heatmaps': self._generate_heatmap_data(),
            'risk_scores': self._calculate_risk_scores(),
            'timestamp': datetime.now().isoformat()
        }
        
        return analysis
    
    def _generate_kpis(self) -> Dict:
        """Generate key performance indicators"""
        total_accidents = len(self.data)
        total_deaths = self.data['deaths'].sum() if 'deaths' in self.data.columns else total_accidents
        
        # Top cause
        if 'cause_category' in self.data.columns:
            top_cause = self.data['cause_category'].value_counts().nlargest(1)
            primary_hazard = top_cause.index[0] if len(top_cause) > 0 else 'Unknown'
        else:
            primary_hazard = 'Unknown'
        
        # Highest risk state
        if 'state' in self.data.columns:
            top_state = self.data['state'].value_counts().nlargest(1)
            highest_risk_state = top_state.index[0] if len(top_state) > 0 else 'Unknown'
        else:
            highest_risk_state = 'Unknown'
        
        return {
            'total_accidents': int(total_accidents),
            'total_deaths': int(total_deaths),
            'primary_hazard': primary_hazard,
            'highest_risk_state': highest_risk_state
        }
    
    def _analyze_trends(self) -> Dict:
        """Analyze temporal and categorical trends"""
        trends = {}
        
        # Accidents by cause
        if 'cause_category' in self.data.columns:
            cause_counts = self.data['cause_category'].value_counts().to_dict()
            trends['by_cause'] = cause_counts
        
        # Accidents by state
        if 'state' in self.data.columns:
            state_counts = self.data['state'].value_counts().to_dict()
            trends['by_state'] = state_counts
        
        # Temporal trends (by year if available)
        if 'year' in self.data.columns:
            year_counts = self.data['year'].value_counts().sort_index().to_dict()
            trends['by_year'] = year_counts
        
        return trends
    
    def _generate_heatmap_data(self) -> List[Dict]:
        """Generate geographic heatmap data"""
        if 'lat' not in self.data.columns or 'lon' not in self.data.columns:
            return []
        
        # Aggregate by location
        location_data = self.data.groupby(['state', 'lat', 'lon']).agg({
            'id': 'count',
            'deaths': 'sum'
        }).reset_index()
        
        location_data.columns = ['state', 'lat', 'lon', 'accidents', 'deaths']
        
        return location_data.to_dict('records')
    
    def _calculate_risk_scores(self) -> Dict:
        """Calculate risk scores by different dimensions"""
        risk_scores = {}
        
        # Risk by state (normalized)
        if 'state' in self.data.columns:
            state_deaths = self.data.groupby('state')['deaths'].sum().sort_values(ascending=False)
            max_deaths = state_deaths.max() if len(state_deaths) > 0 else 1
            risk_scores['by_state'] = {
                state: float(deaths / max_deaths * 100) 
                for state, deaths in state_deaths.head(10).items()
            }
        
        # Risk by cause
        if 'cause_category' in self.data.columns:
            cause_deaths = self.data.groupby('cause_category')['deaths'].sum().sort_values(ascending=False)
            max_deaths = cause_deaths.max() if len(cause_deaths) > 0 else 1
            risk_scores['by_cause'] = {
                cause: float(deaths / max_deaths * 100)
                for cause, deaths in cause_deaths.head(10).items()
            }
        
        return risk_scores
    
    def _empty_analysis(self) -> Dict:
        """Return empty analysis structure"""
        return {
            'kpis': {
                'total_accidents': 0,
                'total_deaths': 0,
                'primary_hazard': 'No Data',
                'highest_risk_state': 'No Data'
            },
            'trends': {},
            'heatmaps': [],
            'risk_scores': {}
        }


class QueryAgent:
    """
    Agent 2: Query Agent
    - Natural language interface for safety inquiries
    - Interprets complex queries
    - Semantic search using embeddings
    """
    
    def __init__(self, data: pd.DataFrame, pathway_stream=None):
        self.data = data
        self.pathway_stream = pathway_stream
        self.name = "Query Agent"
    
    def process_query(self, query: str) -> Dict:
        """Process natural language query"""
        
        query_lower = query.lower().strip()
        
        # Determine query type
        query_type = self._classify_query(query_lower)
        
        if query_type == 'summary':
            return self._generate_summary()
        elif query_type == 'entity':
            return self._entity_search(query_lower)
        elif query_type == 'keyword':
            return self._keyword_search(query_lower)
        elif query_type == 'semantic':
            return self._semantic_search(query_lower)
        else:
            return self._general_search(query_lower)
    
    def _classify_query(self, query: str) -> str:
        """Classify the type of query"""
        summary_keywords = ['summary', 'summarise', 'summarize', 'overview', 'tldr']
        
        if any(kw in query for kw in summary_keywords):
            return 'summary'
        
        # Check if query mentions specific mine name
        if 'mine_name_lower' in self.data.columns:
            for mine in self.data['mine_name_lower'].dropna().unique():
                if len(mine) > 5 and mine.lower() in query:
                    return 'entity'
        
        # Check for specific causes or states
        if 'cause_category' in self.data.columns:
            for cause in self.data['cause_category'].dropna().unique():
                if cause.lower() in query:
                    return 'keyword'
        
        return 'semantic'
    
    def _generate_summary(self) -> Dict:
        """Generate overall summary"""
        if len(self.data) == 0:
            return {
                'type': 'summary',
                'message': 'No accident data available.',
                'data': None
            }
        
        total_accidents = len(self.data)
        total_deaths = self.data['deaths'].sum() if 'deaths' in self.data.columns else total_accidents
        
        # Top causes
        top_causes = []
        if 'cause_category' in self.data.columns:
            cause_counts = self.data['cause_category'].value_counts().nlargest(3)
            top_causes = [
                {'cause': cause, 'count': int(count)}
                for cause, count in cause_counts.items()
            ]
        
        # Top states
        top_states = []
        if 'state' in self.data.columns:
            state_counts = self.data['state'].value_counts().nlargest(3)
            top_states = [
                {'state': state, 'count': int(count)}
                for state, count in state_counts.items()
            ]
        
        return {
            'type': 'summary',
            'message': f'Analysis of {total_accidents} fatal accidents resulting in {total_deaths} deaths.',
            'data': {
                'total_accidents': int(total_accidents),
                'total_deaths': int(total_deaths),
                'top_causes': top_causes,
                'top_states': top_states
            }
        }
    
    def _entity_search(self, query: str) -> Dict:
        """Search for specific entity (mine, location)"""
        # Clean query
        preambles = ['tell me about', 'what is', 'what happened at', 'info on', 'details for']
        for preamble in preambles:
            if query.startswith(preamble):
                query = query[len(preamble):].strip()
        
        # Search in mine names
        if 'mine_name_lower' in self.data.columns:
            matches = self.data[self.data['mine_name_lower'].str.contains(query, na=False)]
            
            if len(matches) > 0:
                results = []
                for _, row in matches.iterrows():
                    results.append({
                        'mine_name': row.get('mine_name', 'Unknown'),
                        'date': row.get('date').strftime('%Y-%m-%d') if pd.notna(row.get('date')) else 'Unknown',
                        'location': f"{row.get('district', 'Unknown')}, {row.get('state', 'Unknown')}",
                        'owner': row.get('owner', 'Unknown'),
                        'cause': row.get('cause_category', 'Unknown'),
                        'victim': row.get('killed_person', 'Unknown'),
                        'description': row.get('description', 'No description available')[:300]
                    })
                
                return {
                    'type': 'entity',
                    'message': f'Found {len(matches)} incident(s) matching "{query}"',
                    'data': results
                }
        
        return {
            'type': 'entity',
            'message': f'No incidents found matching "{query}"',
            'data': []
        }
    
    def _keyword_search(self, query: str) -> Dict:
        """Keyword-based search"""
        if 'raw_text_for_search' not in self.data.columns:
            return {'type': 'keyword', 'message': 'Search not available', 'data': []}
        
        matches = self.data[self.data['raw_text_for_search'].str.contains(query, na=False)]
        
        if len(matches) > 0:
            # Generate analysis
            top_cause = matches['cause_category'].mode()[0] if 'cause_category' in matches.columns and len(matches) > 0 else 'Unknown'
            top_state = matches['state'].mode()[0] if 'state' in matches.columns and len(matches) > 0 else 'Unknown'
            
            results = []
            for _, row in matches.head(10).iterrows():
                results.append({
                    'date': row.get('date').strftime('%Y-%m-%d') if pd.notna(row.get('date')) else 'Unknown',
                    'mine_name': row.get('mine_name', 'Unknown'),
                    'location': f"{row.get('district', 'Unknown')}, {row.get('state', 'Unknown')}",
                    'cause': row.get('cause_category', 'Unknown'),
                    'description': row.get('description', 'No description')[:200]
                })
            
            return {
                'type': 'keyword',
                'message': f'Found {len(matches)} incidents. Primary hazard: {top_cause}, High concentration in: {top_state}',
                'data': results,
                'analysis': {
                    'total_matches': int(len(matches)),
                    'primary_hazard': top_cause,
                    'high_risk_state': top_state
                }
            }
        
        return {
            'type': 'keyword',
            'message': f'No incidents found for query: "{query}"',
            'data': []
        }
    
    def _semantic_search(self, query: str) -> Dict:
        """Semantic search using embeddings"""
        if self.pathway_stream is None:
            return self._keyword_search(query)
        
        # Get semantic matches
        indices = self.pathway_stream.semantic_search(query, top_k=5)
        
        if len(indices) > 0:
            matches = self.data.iloc[indices]
            
            results = []
            for _, row in matches.iterrows():
                results.append({
                    'date': row.get('date').strftime('%Y-%m-%d') if pd.notna(row.get('date')) else 'Unknown',
                    'mine_name': row.get('mine_name', 'Unknown'),
                    'cause': row.get('cause_category', 'Unknown'),
                    'description': row.get('description', '')[:200]
                })
            
            return {
                'type': 'semantic',
                'message': f'Found {len(matches)} semantically related incidents',
                'data': results
            }
        
        return {'type': 'semantic', 'message': 'No matches found', 'data': []}
    
    def _general_search(self, query: str) -> Dict:
        """General fallback search"""
        return self._keyword_search(query)


class AuditAgent:
    """
    Agent 3: Audit Agent
    - Cross-references with DGMS regulations
    - Produces automated safety audit reports
    - Flags probable regulatory violations
    """
    
    def __init__(self, data: pd.DataFrame, regulatory_kb):
        self.data = data
        self.regulatory_kb = regulatory_kb
        self.name = "Audit Agent"
    
    def generate_audit(self) -> Dict:
        """Generate comprehensive safety audit report"""
        
        if len(self.data) == 0:
            return self._empty_audit()
        
        audit = {
            'executive_summary': self._generate_executive_summary(),
            'hazard_analysis': self._analyze_hazards(),
            'compliance_check': self._check_compliance(),
            'recommendations': self._generate_recommendations(),
            'report_id': f"AGENT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'generated_at': datetime.now().isoformat()
        }
        
        return audit
    
    def _generate_executive_summary(self) -> Dict:
        """Generate executive summary"""
        total_accidents = len(self.data)
        total_deaths = self.data['deaths'].sum() if 'deaths' in self.data.columns else total_accidents
        
        # Date range
        if 'date' in self.data.columns:
            date_range = f"{self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}"
        else:
            date_range = "Unknown"
        
        # Top hazards
        top_hazards = []
        if 'cause_category' in self.data.columns:
            top_hazards = self.data['cause_category'].value_counts().head(3).to_dict()
        
        return {
            'total_accidents': int(total_accidents),
            'total_deaths': int(total_deaths),
            'date_range': date_range,
            'top_hazards': top_hazards
        }
    
    def _analyze_hazards(self) -> List[Dict]:
        """Analyze hazard patterns and root causes"""
        hazard_analysis = []
        
        if 'cause_category' not in self.data.columns:
            return hazard_analysis
        
        # Group by cause category
        cause_groups = self.data.groupby('cause_category').agg({
            'deaths': 'sum',
            'id': 'count'
        }).reset_index()
        
        cause_groups.columns = ['cause', 'deaths', 'incidents']
        cause_groups = cause_groups.sort_values('deaths', ascending=False)
        
        for _, row in cause_groups.iterrows():
            cause = row['cause']
            deaths = int(row['deaths'])
            incidents = int(row['incidents'])
            
            # Get regulation
            regulation = self.regulatory_kb.get_violated_regulation(cause)
            
            # Get root cause analysis
            root_cause = self._identify_root_cause(cause)
            
            hazard_analysis.append({
                'hazard': cause,
                'deaths': deaths,
                'incidents': incidents,
                'regulation_violated': regulation,
                'root_cause': root_cause,
                'severity': 'Critical' if deaths > 5 else 'High' if deaths > 2 else 'Medium'
            })
        
        return hazard_analysis
    
    def _identify_root_cause(self, cause_category: str) -> str:
        """Identify root cause for hazard category"""
        root_causes = {
            'Fall of Roof': 'Inadequate roof support and failure to comply with support regulations',
            'Fall of Sides': 'Improper benching/sloping and inadequate securing of quarry sides',
            'Dumpers': 'Brake system failures, operator negligence, and inadequate haul road maintenance',
            'Fall of Person': 'Failure to provide or enforce use of safety belts for work at heights',
            'Explosives': 'Improper explosive handling procedures and inadequate safety protocols',
            'Electricity': 'Defective electrical installations and lack of lockout-tagout procedures',
            'Conveyors': 'Absence of machine guarding and lockout-tagout during maintenance',
        }
        
        return root_causes.get(cause_category, 'Systemic safety management failures')
    
    def _check_compliance(self) -> List[Dict]:
        """Check regulatory compliance"""
        compliance_issues = []
        
        if 'cause_category' not in self.data.columns:
            return compliance_issues
        
        # Identify violated regulations
        for cause in self.data['cause_category'].dropna().unique():
            regulation_id = self.regulatory_kb.get_violated_regulation(cause)
            
            if regulation_id:
                # Count violations
                violation_count = len(self.data[self.data['cause_category'] == cause])
                
                regulation_details = self.regulatory_kb.get_regulation_details(regulation_id)
                
                compliance_issues.append({
                    'regulation': regulation_id,
                    'title': regulation_details['title'] if regulation_details else 'Unknown',
                    'violation_count': int(violation_count),
                    'hazard_type': cause,
                    'status': 'Non-Compliant',
                    'priority': 'High' if violation_count > 3 else 'Medium'
                })
        
        return sorted(compliance_issues, key=lambda x: x['violation_count'], reverse=True)
    
    def _generate_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations"""
        recommendations = []
        
        # Recommendations based on top hazards
        if 'cause_category' in self.data.columns:
            top_causes = self.data['cause_category'].value_counts().head(3)
            
            recommendation_templates = {
                'Fall of Roof': {
                    'action': 'Mandate immediate roof support audits in all underground operations',
                    'regulation': 'MMR 106(1)',
                    'priority': 'Critical'
                },
                'Fall of Sides': {
                    'action': 'Audit slope stability and benching in all opencast mines',
                    'regulation': 'MMR 106(3)',
                    'priority': 'Critical'
                },
                'Dumpers': {
                    'action': 'Implement mandatory pre-shift brake system checks for all haulage vehicles',
                    'regulation': 'MMR 181',
                    'priority': 'High'
                },
                'Fall of Person': {
                    'action': 'Prohibit work above 1.8m without approved safety harness',
                    'regulation': 'MMR 118(4)',
                    'priority': 'Critical'
                },
                'Electricity': {
                    'action': 'Conduct electrical safety audits and implement lockout-tagout',
                    'regulation': 'MMR 127',
                    'priority': 'High'
                }
            }
            
            for cause in top_causes.index:
                if cause in recommendation_templates:
                    recommendations.append(recommendation_templates[cause])
        
        return recommendations
    
    def _empty_audit(self) -> Dict:
        """Return empty audit structure"""
        return {
            'executive_summary': {
                'total_accidents': 0,
                'total_deaths': 0,
                'date_range': 'No Data'
            },
            'hazard_analysis': [],
            'compliance_check': [],
            'recommendations': []
        }


class SARSAAgentOrchestrator:
    """
    Orchestrates all three agents using LangGraph (or fallback)
    """
    
    def __init__(self, data: pd.DataFrame, pathway_stream=None, regulatory_kb=None):
        self.data = data
        self.analysis_agent = AnalysisAgent(data)
        self.query_agent = QueryAgent(data, pathway_stream)
        self.audit_agent = AuditAgent(data, regulatory_kb)
        
        if LANGGRAPH_AVAILABLE:
            self.graph = self._build_langgraph()
        else:
            self.graph = None
    
    def _build_langgraph(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(AgentState)
        
        # Define nodes (agents)
        workflow.add_node("analysis", self._analysis_node)
        workflow.add_node("query", self._query_node)
        workflow.add_node("audit", self._audit_node)
        
        # Define edges (simplified for this demo)
        workflow.set_entry_point("query")
        workflow.add_edge("query", END)
        workflow.add_edge("analysis", END)
        workflow.add_edge("audit", END)
        
        return workflow.compile()
    
    def _analysis_node(self, state: AgentState):
        """Analysis agent node"""
        result = self.analysis_agent.analyze()
        return {"result": result, "agent_history": [self.analysis_agent.name]}
    
    def _query_node(self, state: AgentState):
        """Query agent node"""
        result = self.query_agent.process_query(state["query"])
        return {"result": result, "agent_history": [self.query_agent.name]}
    
    def _audit_node(self, state: AgentState):
        """Audit agent node"""
        result = self.audit_agent.generate_audit()
        return {"result": result, "agent_history": [self.audit_agent.name]}
    
    def run_analysis(self) -> Dict:
        """Run analysis agent"""
        return self.analysis_agent.analyze()
    
    def run_query(self, query: str) -> Dict:
        """Run query agent"""
        return self.query_agent.process_query(query)
    
    def run_audit(self) -> Dict:
        """Run audit agent"""
        return self.audit_agent.generate_audit()
