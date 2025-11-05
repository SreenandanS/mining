"""
SARSA Agentic AI - LangGraph Implementation (OpenAI conversational layer)
Three intelligent agents: Analysis Agent, Query Agent, Audit Agent
- Uses ChatOpenAI from langchain_openai with your OPENAI_API_KEY in env
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
import pandas as pd
from datetime import datetime
import json
import operator
import os

# LangGraph / LangChain imports
try:
    from langgraph.graph import StateGraph, END
    from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
    from langchain_openai import ChatOpenAI
    LANGGRAPH_AVAILABLE = True
except Exception as e:
    LANGGRAPH_AVAILABLE = False
    _IMPORT_ERR = e
    print("LangGraph/LangChain not available - using fallback implementation")

class AgentState(TypedDict):
    """State shared across all agents"""
    messages: Annotated[List, operator.add]
    data: pd.DataFrame
    query: str
    result: Dict
    agent_history: List[str]


# ---------------------------
# Conversationalizer (OpenAI)
# ---------------------------
class LLMConversationalizer:
    """
    Wraps ChatOpenAI for converting structured results to friendly natural language.
    Requires OPENAI_API_KEY in environment.
    """
    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.2):
        self.model = ChatOpenAI(model=model, temperature=temperature)

    def render(self, user_query: str, structured_result: Dict) -> str:
        def compact(obj, max_chars=2000):
            try:
                s = json.dumps(obj, ensure_ascii=False, default=str)
            except Exception:
                s = str(obj)
            return s[:max_chars]

        system_prompt = (
            "You are SARSA, a helpful mine-safety assistant for Indian DGMS accident data. "
            "Speak concisely in clear, conversational English, and ground your answers ONLY in the provided context. "
            "If details are missing, say so briefly. Prefer bullets for lists. Avoid hallucinations."
        )
        user_prompt = (
            f"User query:\n{user_query}\n\n"
            f"Structured result (JSON):\n{compact(structured_result)}"
        )

        msgs = [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        out = self.model.invoke(msgs)
        return out.content if hasattr(out, "content") else str(out)


class AnalysisAgent:
    """Agent 1: generates KPIs, trends, heatmaps, risk scores"""
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.name = "Analysis Agent"

    def analyze(self) -> Dict:
        if len(self.data) == 0:
            return self._empty_analysis()
        return {
            'kpis': self._generate_kpis(),
            'trends': self._analyze_trends(),
            'heatmaps': self._generate_heatmap_data(),
            'risk_scores': self._calculate_risk_scores(),
            'timestamp': datetime.now().isoformat()
        }

    def _generate_kpis(self) -> Dict:
        total_accidents = len(self.data)
        total_deaths = int(self.data['deaths'].sum()) if 'deaths' in self.data.columns else total_accidents
        primary_hazard = 'Unknown'
        if 'cause_category' in self.data.columns and self.data['cause_category'].notna().any():
            vc = self.data['cause_category'].value_counts()
            if len(vc) > 0:
                primary_hazard = vc.index[0]
        highest_risk_state = 'Unknown'
        if 'state' in self.data.columns and self.data['state'].notna().any():
            vc = self.data['state'].value_counts()
            if len(vc) > 0:
                highest_risk_state = vc.index[0]
        return {
            'total_accidents': int(total_accidents),
            'total_deaths': int(total_deaths),
            'primary_hazard': primary_hazard,
            'highest_risk_state': highest_risk_state
        }

    def _analyze_trends(self) -> Dict:
        trends = {}
        if 'cause_category' in self.data.columns:
            trends['by_cause'] = self.data['cause_category'].value_counts().to_dict()
        if 'state' in self.data.columns:
            trends['by_state'] = self.data['state'].value_counts().to_dict()
        if 'year' in self.data.columns:
            trends['by_year'] = self.data['year'].value_counts().sort_index().to_dict()
        return trends

    def _generate_heatmap_data(self) -> List[Dict]:
        if 'lat' not in self.data.columns or 'lon' not in self.data.columns:
            return []
        gd = self.data.groupby(['state', 'lat', 'lon']).agg({'id':'count','deaths':'sum'}).reset_index()
        gd.columns = ['state','lat','lon','accidents','deaths']
        return gd.to_dict('records')

    def _calculate_risk_scores(self) -> Dict:
        risk_scores = {}
        if 'state' in self.data.columns and 'deaths' in self.data.columns:
            state_deaths = self.data.groupby('state')['deaths'].sum().sort_values(ascending=False)
            maxd = max(int(state_deaths.max()), 1) if len(state_deaths)>0 else 1
            risk_scores['by_state'] = {s: float(d/maxd*100) for s,d in state_deaths.head(10).items()}
        if 'cause_category' in self.data.columns and 'deaths' in self.data.columns:
            cause_deaths = self.data.groupby('cause_category')['deaths'].sum().sort_values(ascending=False)
            maxd = max(int(cause_deaths.max()), 1) if len(cause_deaths)>0 else 1
            risk_scores['by_cause'] = {c: float(d/maxd*100) for c,d in cause_deaths.head(10).items()}
        return risk_scores

    def _empty_analysis(self) -> Dict:
        return {
            'kpis': {'total_accidents':0,'total_deaths':0,'primary_hazard':'No Data','highest_risk_state':'No Data'},
            'trends': {}, 'heatmaps': [], 'risk_scores': {}
        }


class QueryAgent:
    """Agent 2: NL query interpreter with keyword/semantic/entity search hooks"""
    def __init__(self, data: pd.DataFrame, pathway_stream=None):
        self.data = data
        self.pathway_stream = pathway_stream  # still used as a generic stream/semantic provider
        self.name = "Query Agent"

    def process_query(self, query: str) -> Dict:
        q = (query or "").lower().strip()
        qtype = self._classify_query(q)
        if qtype == 'summary':
            return self._generate_summary()
        if qtype == 'entity':
            return self._entity_search(q)
        if qtype == 'keyword':
            return self._keyword_search(q)
        return self._semantic_search(q)

    def _classify_query(self, q: str) -> str:
        if any(k in q for k in ['summary','summarize','summarise','overview','tldr']):
            return 'summary'
        if 'mine_name_lower' in self.data.columns:
            for mine in self.data['mine_name_lower'].dropna().unique():
                if isinstance(mine, str) and len(mine)>5 and mine in q:
                    return 'entity'
        if 'cause_category' in self.data.columns:
            for cause in self.data['cause_category'].dropna().unique():
                if isinstance(cause, str) and cause.lower() in q:
                    return 'keyword'
        return 'semantic'

    def _generate_summary(self) -> Dict:
        if len(self.data)==0:
            return {'type':'summary','message':'No accident data available.','data':None}
        total_accidents = len(self.data)
        total_deaths = int(self.data['deaths'].sum()) if 'deaths' in self.data.columns else total_accidents
        top_causes, top_states = [], []
        if 'cause_category' in self.data.columns:
            cc = self.data['cause_category'].value_counts().nlargest(3)
            top_causes = [{'cause':c,'count':int(n)} for c,n in cc.items()]
        if 'state' in self.data.columns:
            sc = self.data['state'].value_counts().nlargest(3)
            top_states = [{'state':s,'count':int(n)} for s,n in sc.items()]
        return {'type':'summary','message':f'Analysis of {total_accidents} accidents with {total_deaths} deaths.',
                'data':{'total_accidents':int(total_accidents),'total_deaths':int(total_deaths),
                        'top_causes':top_causes,'top_states':top_states}}

    def _entity_search(self, q: str) -> Dict:
        pre = ['tell me about','what is','what happened at','info on','details for']
        for p in pre:
            if q.startswith(p):
                q = q[len(p):].strip()
        if 'mine_name_lower' in self.data.columns:
            m = self.data[self.data['mine_name_lower'].str.contains(q, na=False)]
            if len(m)>0:
                data = []
                for _,row in m.iterrows():
                    data.append({
                        'mine_name': row.get('mine_name','Unknown'),
                        'date': row.get('date').strftime('%Y-%m-%d') if pd.notna(row.get('date')) else 'Unknown',
                        'location': f"{row.get('district','Unknown')}, {row.get('state','Unknown')}",
                        'owner': row.get('owner','Unknown'),
                        'cause': row.get('cause_category','Unknown'),
                        'victim': row.get('killed_person','Unknown'),
                        'description': (row.get('description','') or '')[:300]
                    })
                return {'type':'entity','message':f'Found {len(m)} incident(s) matching \"{q}\"','data':data}
        return {'type':'entity','message':f'No incidents found matching \"{q}\"','data':[]}

    def _keyword_search(self, q: str) -> Dict:
        if 'raw_text_for_search' not in self.data.columns:
            return {'type':'keyword','message':'Search not available','data':[]}
        m = self.data[self.data['raw_text_for_search'].str.contains(q, na=False)]
        if len(m)>0:
            top_cause = m['cause_category'].mode()[0] if 'cause_category' in m.columns and len(m)>0 else 'Unknown'
            top_state = m['state'].mode()[0] if 'state' in m.columns and len(m)>0 else 'Unknown'
            data = []
            for _,row in m.head(10).iterrows():
                data.append({
                    'date': row.get('date').strftime('%Y-%m-%d') if pd.notna(row.get('date')) else 'Unknown',
                    'mine_name': row.get('mine_name','Unknown'),
                    'location': f"{row.get('district','Unknown')}, {row.get('state','Unknown')}",
                    'cause': row.get('cause_category','Unknown'),
                    'description': (row.get('description','') or '')[:200]
                })
            return {'type':'keyword','message':f'Found {len(m)} incidents. Primary hazard: {top_cause}, Concentrated in: {top_state}',
                    'data':data, 'analysis':{'total_matches':int(len(m)),'primary_hazard':top_cause,'high_risk_state':top_state}}
        return {'type':'keyword','message':f'No incidents found for \"{q}\"','data':[]}

    def _semantic_search(self, q: str) -> Dict:
        if self.pathway_stream is None:
            return self._keyword_search(q)
        idxs = self.pathway_stream.semantic_search(q, top_k=5)
        if len(idxs)>0:
            matches = self.data.iloc[idxs]
            data = []
            for _,row in matches.iterrows():
                data.append({
                    'date': row.get('date').strftime('%Y-%m-%d') if pd.notna(row.get('date')) else 'Unknown',
                    'mine_name': row.get('mine_name','Unknown'),
                    'cause': row.get('cause_category','Unknown'),
                    'description': (row.get('description','') or '')[:400]
                })
            rag_ctx = self.pathway_stream.build_rag_context(idxs)
            return {'type':'semantic','message':f'Found {len(matches)} semantically related incidents','data':data,'rag_context':rag_ctx}
        return {'type':'semantic','message':'No matches found','data':[]}


class AuditAgent:
    """Agent 3: compliance & recommendations"""
    def __init__(self, data: pd.DataFrame, regulatory_kb):
        self.data = data
        self.regulatory_kb = regulatory_kb
        self.name = "Audit Agent"

    def generate_audit(self) -> Dict:
        if len(self.data)==0:
            return self._empty_audit()
        return {
            'executive_summary': self._generate_executive_summary(),
            'hazard_analysis': self._analyze_hazards(),
            'compliance_check': self._check_compliance(),
            'recommendations': self._generate_recommendations(),
            'report_id': f"AGENT-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'generated_at': datetime.now().isoformat()
        }

    def _generate_executive_summary(self) -> Dict:
        total_accidents = len(self.data)
        total_deaths = int(self.data['deaths'].sum()) if 'deaths' in self.data.columns else total_accidents
        if 'date' in self.data.columns and self.data['date'].notna().any():
            dr = f"{self.data['date'].min().strftime('%Y-%m-%d')} to {self.data['date'].max().strftime('%Y-%m-%d')}"
        else:
            dr = "Unknown"
        top_hazards = {}
        if 'cause_category' in self.data.columns:
            top_hazards = self.data['cause_category'].value_counts().head(3).to_dict()
        return {'total_accidents':int(total_accidents),'total_deaths':int(total_deaths),'date_range':dr,'top_hazards':top_hazards}

    def _analyze_hazards(self) -> List[Dict]:
        out = []
        if 'cause_category' not in self.data.columns:
            return out
        g = self.data.groupby('cause_category').agg({'deaths':'sum','id':'count'}).reset_index()
        g.columns = ['cause','deaths','incidents']
        g = g.sort_values('deaths', ascending=False)
        for _,r in g.iterrows():
            cause = r['cause']; deaths=int(r['deaths']); incidents=int(r['incidents'])
            regulation = self.regulatory_kb.get_violated_regulation(cause)
            root_cause = self._identify_root_cause(cause)
            out.append({'hazard':cause,'deaths':deaths,'incidents':incidents,
                        'regulation_violated':regulation,'root_cause':root_cause,
                        'severity':'Critical' if deaths>5 else 'High' if deaths>2 else 'Medium'})
        return out

    def _identify_root_cause(self, cause_category: str) -> str:
        root_causes = {
            'Fall of Roof':'Inadequate roof support and failure to comply with support regulations',
            'Fall of Sides':'Improper benching/sloping and inadequate securing of quarry sides',
            'Dumpers':'Brake failures, operator error, poor haul-road maintenance',
            'Fall of Person':'Lack of harnesses/edge protection for work at height',
            'Explosives':'Improper handling, storage, or blasting procedure',
            'Electricity':'Defective installations and missing lockout-tagout',
            'Conveyors':'Missing guards and maintenance without isolation',
        }
        return root_causes.get(cause_category, 'Systemic safety management failures')

    def _check_compliance(self) -> List[Dict]:
        issues = []
        if 'cause_category' not in self.data.columns:
            return issues
        for cause in self.data['cause_category'].dropna().unique():
            reg = self.regulatory_kb.get_violated_regulation(cause)
            if reg:
                count = int((self.data['cause_category']==cause).sum())
                details = self.regulatory_kb.get_regulation_details(reg) or {}
                issues.append({'regulation':reg,'title':details.get('title','Unknown'),
                               'violation_count':count,'hazard_type':cause,'status':'Non-Compliant',
                               'priority':'High' if count>3 else 'Medium'})
        return sorted(issues, key=lambda x: x['violation_count'], reverse=True)

    def _generate_recommendations(self) -> List[Dict]:
        recs = []
        if 'cause_category' in self.data.columns:
            top = self.data['cause_category'].value_counts().head(3)
            templates = {
                'Fall of Roof': {'action':'Immediate roof support audits in all UG operations','regulation':'MMR 106(1)','priority':'Critical'},
                'Fall of Sides': {'action':'Audit slope stability/bench design in opencast mines','regulation':'MMR 106(3)','priority':'Critical'},
                'Dumpers': {'action':'Mandatory pre-shift brake checks for all haulage vehicles','regulation':'MMR 181','priority':'High'},
                'Fall of Person': {'action':'No work above 1.8 m without approved harness','regulation':'MMR 118(4)','priority':'Critical'},
                'Electricity': {'action':'Electrical audits + lockout-tagout enforcement','regulation':'MMR 127','priority':'High'}
            }
            for c in top.index:
                if c in templates: recs.append(templates[c])
        return recs

    def _empty_audit(self) -> Dict:
        return {'executive_summary':{'total_accidents':0,'total_deaths':0,'date_range':'No Data'},
                'hazard_analysis':[],'compliance_check':[],'recommendations':[]}


class SARSAAgentOrchestrator:
    """Orchestrates three agents; provides natural-language conversion via OpenAI."""
    def __init__(self, data: pd.DataFrame, pathway_stream=None, regulatory_kb=None, conversationalizer: Optional[LLMConversationalizer]=None):
        self.data = data
        self.analysis_agent = AnalysisAgent(data)
        self.query_agent = QueryAgent(data, pathway_stream)
        self.audit_agent = AuditAgent(data, regulatory_kb)
        self.pathway_stream = pathway_stream
        self.conv = conversationalizer or LLMConversationalizer()

        self.graph = self._build_langgraph() if LANGGRAPH_AVAILABLE else None

    def _build_langgraph(self):
        wf = StateGraph(AgentState)
        wf.add_node("analysis", self._analysis_node)
        wf.add_node("query", self._query_node)
        wf.add_node("audit", self._audit_node)
        wf.set_entry_point("query")
        wf.add_edge("query", END)
        wf.add_edge("analysis", END)
        wf.add_edge("audit", END)
        return wf.compile()

    def _analysis_node(self, state: AgentState):
        result = self.analysis_agent.analyze()
        return {"result": result, "agent_history": [self.analysis_agent.name]}

    def _query_node(self, state: AgentState):
        result = self.query_agent.process_query(state["query"])
        return {"result": result, "agent_history": [self.query_agent.name]}

    def _audit_node(self, state: AgentState):
        result = self.audit_agent.generate_audit()
        return {"result": result, "agent_history": [self.audit_agent.name]}

    def run_analysis(self) -> Dict:
        return self.analysis_agent.analyze()

    def run_query(self, query: str) -> Dict:
        return self.query_agent.process_query(query)

    def run_audit(self) -> Dict:
        return self.audit_agent.generate_audit()

    def to_natural_language(self, user_query: str, structured_result: Dict) -> str:
        return self.conv.render(user_query, structured_result)
