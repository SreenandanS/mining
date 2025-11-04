"""
Multi-Agent System with Specialized Agents for Mining Safety
Implements collaborative agent architecture using LangGraph
"""

from typing import TypedDict, Annotated, Sequence, List, Dict, Optional
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage, AIMessage, SystemMessage
from langgraph.graph import StateGraph, END
import operator
import json
from datetime import datetime
import os


class AgentState(TypedDict):
    """Enhanced state for multi-agent collaboration"""
    messages: Annotated[Sequence[HumanMessage | AIMessage], operator.add]
    query: str
    rag_results: List[Dict]
    
    # Analysis outputs
    classification: Dict
    risk_assessment: Dict
    pattern_analysis: Dict
    compliance_check: Dict
    
    # Recommendations and alerts
    immediate_actions: List[str]
    long_term_recommendations: List[str]
    training_needs: List[str]
    hazard_alerts: List[str]
    
    # Agent coordination
    current_agent: str
    agent_outputs: Dict


class InspectorAgent:
    """Specialized agent for incident inspection and classification"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "Inspector"
    
    def analyze(self, state: AgentState) -> Dict:
        """Inspect and classify incidents with detailed analysis"""
        rag_results = state["rag_results"]
        
        if not rag_results:
            return {"error": "No data available", "agent": self.name}
        
        # Combine context with metadata
        context_parts = []
        for result in rag_results[:3]:
            content = result["content"][:600]
            metadata = result.get("metadata", {})
            context_parts.append(
                f"Document: {metadata.get('source', 'unknown')}\n"
                f"Year: {metadata.get('year', 'unknown')}\n"
                f"Content: {content}"
            )
        
        context = "\n\n---\n\n".join(context_parts)
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are an expert Mining Safety Inspector with 20 years of experience.
            
            Analyze the incident reports and provide a detailed classification:
            
            **Required Output (JSON format):**
            {
                "incident_type": "primary type (roof_fall, explosion, machinery, transportation, methane, electrical, fire)",
                "secondary_types": ["list", "of", "contributing", "factors"],
                "severity": "fatal|serious|minor",
                "location": {
                    "state": "state name",
                    "mine_type": "underground|opencast",
                    "specific_area": "where in mine"
                },
                "casualties": {
                    "fatalities": number,
                    "injuries": number,
                    "affected": number
                },
                "immediate_cause": "direct cause description",
                "contributing_factors": ["factor1", "factor2"],
                "equipment_involved": ["list of equipment"],
                "time_context": "when incident occurred",
                "regulatory_violations": ["if any violations detected"]
            }
            
            Be specific and evidence-based. Extract actual data from the documents."""),
            HumanMessage(content=f"Query: {state['query']}\n\n=== INCIDENT REPORTS ===\n{context}")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            classification = self._parse_json_response(response.content)
            classification["inspector_agent"] = self.name
            return classification
        except Exception as e:
            return {"error": str(e), "agent": self.name}
    
    def _parse_json_response(self, content: str) -> Dict:
        """Extract JSON from response"""
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
            return {"raw_response": content}
        except:
            return {"raw_response": content}


class ComplianceAgent:
    """Specialized agent for regulatory compliance checking"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "Compliance Officer"
        
        # DGMS regulation knowledge base
        self.regulations = {
            "ventilation": "CMR 2017 Chapter VI - Ventilation and Gas Testing",
            "roof_support": "CMR 2017 Chapter VII - Support and Roof Control",
            "transport": "CMR 2017 Chapter XI - Haulage and Transport",
            "electricity": "CMR 2017 Chapter IX - Electrical Installations",
            "explosives": "CMR 2017 Chapter VIII - Use of Explosives",
            "machinery": "CMR 2017 Chapter X - Machinery and Equipment"
        }
    
    def analyze(self, state: AgentState) -> Dict:
        """Check compliance against DGMS regulations"""
        classification = state.get("classification", {})
        rag_results = state["rag_results"]
        
        incident_type = classification.get("incident_type", "unknown")
        
        # Get relevant regulation
        relevant_regs = []
        for key, reg in self.regulations.items():
            if key in incident_type.lower() or key in state["query"].lower():
                relevant_regs.append(reg)
        
        context = "\n\n".join([r["content"][:400] for r in rag_results[:2]])
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content=f"""You are a DGMS Compliance Officer expert in Coal Mines Regulations (CMR) 2017.
            
            Analyze the incident for compliance violations and regulatory gaps.
            
            Relevant Regulations: {', '.join(relevant_regs) if relevant_regs else 'General Safety Regulations'}
            
            **Required Output (JSON):**
            {{
                "violations_detected": ["specific regulation violations"],
                "compliance_gaps": ["areas of non-compliance"],
                "mandatory_actions": ["required actions per regulations"],
                "inspection_requirements": ["what needs inspection"],
                "reporting_obligations": ["required reports to DGMS"],
                "penalty_risk": "low|medium|high",
                "corrective_timeline": "immediate|30days|90days"
            }}
            
            Be specific about regulation numbers and requirements."""),
            HumanMessage(content=f"Classification: {json.dumps(classification)}\n\nContext: {context}")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            compliance = self._parse_json_response(response.content)
            compliance["compliance_agent"] = self.name
            return compliance
        except Exception as e:
            return {"error": str(e), "agent": self.name}
    
    def _parse_json_response(self, content: str) -> Dict:
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
            return {"raw_response": content}
        except:
            return {"raw_response": content}


class SafetyAnalystAgent:
    """Specialized agent for pattern analysis and risk assessment"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "Safety Analyst"
    
    def analyze(self, state: AgentState) -> Dict:
        """Perform deep pattern and risk analysis"""
        classification = state.get("classification", {})
        rag_results = state["rag_results"]
        
        # Aggregate metadata for pattern detection
        metadata_summary = self._aggregate_metadata(rag_results)
        
        context = "\n\n".join([r["content"][:500] for r in rag_results[:4]])
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Mining Safety Data Analyst specializing in pattern recognition and risk assessment.
            
            Analyze incident patterns, trends, and risk factors.
            
            **Required Output (JSON):**
            {
                "risk_level": "critical|high|medium|low",
                "risk_factors": [
                    {"factor": "name", "severity": "high|medium|low", "evidence": "why"}
                ],
                "patterns_detected": [
                    {"pattern": "description", "frequency": "how often", "locations": ["where"]}
                ],
                "trend_analysis": {
                    "increasing_risks": ["risk types increasing"],
                    "decreasing_risks": ["risk types decreasing"],
                    "emerging_concerns": ["new patterns"]
                },
                "comparative_analysis": {
                    "similar_incidents": number,
                    "worst_case_scenario": "description",
                    "best_practice_deviation": "what went wrong"
                },
                "predictive_indicators": ["early warning signs for future incidents"]
            }
            
            Focus on actionable insights backed by data."""),
            HumanMessage(content=f"Query: {state['query']}\n\nClassification: {json.dumps(classification)}\n\n"
                                f"Metadata Summary: {json.dumps(metadata_summary)}\n\nContext: {context}")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            analysis = self._parse_json_response(response.content)
            analysis["analyst_agent"] = self.name
            return analysis
        except Exception as e:
            return {"error": str(e), "agent": self.name}
    
    def _aggregate_metadata(self, results: List[Dict]) -> Dict:
        """Aggregate metadata for pattern analysis"""
        years = []
        states = []
        severities = []
        types = []
        
        for result in results:
            meta = result.get("metadata", {})
            if meta.get("year"): years.append(meta["year"])
            if meta.get("state"): states.append(meta["state"])
            if meta.get("severity"): severities.append(meta["severity"])
            if meta.get("incident_types"): types.extend(meta["incident_types"])
        
        return {
            "years": list(set(years)),
            "states": list(set(states)),
            "severity_distribution": {s: severities.count(s) for s in set(severities)},
            "common_types": list(set(types))
        }
    
    def _parse_json_response(self, content: str) -> Dict:
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
            return {"raw_response": content}
        except:
            return {"raw_response": content}


class TrainingCoordinatorAgent:
    """Specialized agent for training and capacity building recommendations"""
    
    def __init__(self, llm):
        self.llm = llm
        self.name = "Training Coordinator"
    
    def analyze(self, state: AgentState) -> Dict:
        """Generate targeted training recommendations"""
        classification = state.get("classification", {})
        risk_assessment = state.get("pattern_analysis", {})
        
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Mining Safety Training Coordinator with expertise in adult learning and safety education.
            
            Based on the incident analysis, design a comprehensive training program.
            
            **Required Output (JSON):**
            {
                "immediate_training": [
                    {"topic": "training need", "target": "who needs it", "urgency": "immediate|1week|1month", "duration": "hours"}
                ],
                "skill_gaps_identified": ["specific skills missing"],
                "refresher_courses": ["topics needing refresher"],
                "certification_requirements": ["certifications needed"],
                "training_methods": {
                    "classroom": ["topics for classroom"],
                    "practical": ["hands-on training needs"],
                    "simulation": ["scenarios for simulation"],
                    "e_learning": ["online modules"]
                },
                "competency_assessments": ["areas to test"],
                "safety_drills": ["drills to conduct"],
                "estimated_cost": "training budget estimate",
                "timeline": "training rollout timeline"
            }
            
            Be specific and practical."""),
            HumanMessage(content=f"Classification: {json.dumps(classification)}\n\n"
                                f"Risk Assessment: {json.dumps(risk_assessment)}")
        ])
        
        try:
            response = self.llm.invoke(prompt.format_messages())
            training = self._parse_json_response(response.content)
            training["training_agent"] = self.name
            return training
        except Exception as e:
            return {"error": str(e), "agent": self.name}
    
    def _parse_json_response(self, content: str) -> Dict:
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start != -1 and end > start:
                return json.loads(content[start:end])
            return {"raw_response": content}
        except:
            return {"raw_response": content}


class MultiAgentSafetySystem:
    """Orchestrates multiple specialized agents for comprehensive analysis"""
    
    def __init__(self, rag_system, openai_api_key: str = None):
        self.rag_system = rag_system
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(temperature=0, model="gpt-4o-mini", openai_api_key=self.openai_api_key)
        
        # Initialize specialized agents
        self.inspector = InspectorAgent(self.llm)
        self.compliance_officer = ComplianceAgent(self.llm)
        self.safety_analyst = SafetyAnalystAgent(self.llm)
        self.training_coordinator = TrainingCoordinatorAgent(self.llm)
        
        self.graph = self._build_graph()
    
    def _build_graph(self) -> StateGraph:
        """Build multi-agent collaboration graph"""
        workflow = StateGraph(AgentState)
        
        # Define nodes for each agent
        workflow.add_node("retrieve", self._retrieve_node)
        workflow.add_node("inspect", self._inspect_node)
        workflow.add_node("analyze_risk", self._analyze_risk_node)
        workflow.add_node("check_compliance", self._check_compliance_node)
        workflow.add_node("plan_training", self._plan_training_node)
        workflow.add_node("generate_alerts", self._generate_alerts_node)
        workflow.add_node("synthesize", self._synthesize_node)
        
        # Define workflow edges
        workflow.set_entry_point("retrieve")
        workflow.add_edge("retrieve", "inspect")
        workflow.add_edge("inspect", "analyze_risk")
        workflow.add_edge("analyze_risk", "check_compliance")
        workflow.add_edge("check_compliance", "plan_training")
        workflow.add_edge("plan_training", "generate_alerts")
        workflow.add_edge("generate_alerts", "synthesize")
        workflow.add_edge("synthesize", END)
        
        return workflow.compile()
    
    def _retrieve_node(self, state: AgentState) -> AgentState:
        """Retrieve relevant documents"""
        query = state["query"]
        results = self.rag_system.query(query, k=6)
        
        state["rag_results"] = results.get("results", [])
        state["current_agent"] = "Retrieval"
        state["messages"].append(AIMessage(content=f"Retrieved {len(state['rag_results'])} documents"))
        return state
    
    def _inspect_node(self, state: AgentState) -> AgentState:
        """Inspector agent classification"""
        state["current_agent"] = self.inspector.name
        classification = self.inspector.analyze(state)
        state["classification"] = classification
        state["agent_outputs"] = {"inspector": classification}
        state["messages"].append(AIMessage(content=f"{self.inspector.name} completed inspection"))
        return state
    
    def _analyze_risk_node(self, state: AgentState) -> AgentState:
        """Safety analyst pattern and risk analysis"""
        state["current_agent"] = self.safety_analyst.name
        analysis = self.safety_analyst.analyze(state)
        state["pattern_analysis"] = analysis
        state["agent_outputs"]["analyst"] = analysis
        state["messages"].append(AIMessage(content=f"{self.safety_analyst.name} completed analysis"))
        return state
    
    def _check_compliance_node(self, state: AgentState) -> AgentState:
        """Compliance officer regulatory check"""
        state["current_agent"] = self.compliance_officer.name
        compliance = self.compliance_officer.analyze(state)
        state["compliance_check"] = compliance
        state["agent_outputs"]["compliance"] = compliance
        state["messages"].append(AIMessage(content=f"{self.compliance_officer.name} completed compliance check"))
        return state
    
    def _plan_training_node(self, state: AgentState) -> AgentState:
        """Training coordinator recommendations"""
        state["current_agent"] = self.training_coordinator.name
        training = self.training_coordinator.analyze(state)
        state["training_needs"] = training.get("immediate_training", [])
        state["agent_outputs"]["training"] = training
        state["messages"].append(AIMessage(content=f"{self.training_coordinator.name} completed training plan"))
        return state
    
    def _generate_alerts_node(self, state: AgentState) -> AgentState:
        """Generate prioritized alerts from all agent outputs"""
        alerts = []
        
        # From classification
        classification = state.get("classification", {})
        if classification.get("severity") == "fatal":
            alerts.append("🚨 CRITICAL: Fatal incident - Immediate DGMS notification required")
        
        # From compliance
        compliance = state.get("compliance_check", {})
        violations = compliance.get("violations_detected", [])
        if violations:
            alerts.append(f"⚠️ COMPLIANCE: {len(violations)} regulatory violations detected")
        
        # From risk analysis
        risk_analysis = state.get("pattern_analysis", {})
        risk_level = risk_analysis.get("risk_level", "unknown")
        if risk_level in ["critical", "high"]:
            alerts.append(f"⚡ RISK ALERT: {risk_level.upper()} risk level - Enhanced monitoring required")
        
        # Pattern-based alerts
        patterns = risk_analysis.get("patterns_detected", [])
        if len(patterns) >= 2:
            alerts.append(f"📊 PATTERN ALERT: {len(patterns)} recurring patterns identified")
        
        state["hazard_alerts"] = alerts if alerts else ["ℹ️ No critical alerts"]
        state["messages"].append(AIMessage(content=f"Generated {len(alerts)} priority alerts"))
        return state
    
    def _synthesize_node(self, state: AgentState) -> AgentState:
        """Synthesize all agent outputs into actionable recommendations"""
        
        # Immediate actions from all agents
        immediate = []
        
        compliance = state.get("compliance_check", {})
        if compliance.get("mandatory_actions"):
            immediate.extend(compliance["mandatory_actions"][:3])
        
        risk = state.get("pattern_analysis", {})
        if risk.get("risk_level") in ["critical", "high"]:
            immediate.append("Immediate site inspection and risk assessment")
        
        # Long-term recommendations
        long_term = []
        
        if risk.get("predictive_indicators"):
            long_term.append("Implement predictive monitoring for identified indicators")
        
        if compliance.get("compliance_gaps"):
            long_term.append("Address systematic compliance gaps through process improvements")
        
        state["immediate_actions"] = immediate[:5]
        state["long_term_recommendations"] = long_term[:5]
        state["messages"].append(AIMessage(content="Synthesized multi-agent recommendations"))
        
        return state
    
    def process_query(self, query: str) -> Dict:
        """Process query through multi-agent system"""
        initial_state = {
            "messages": [HumanMessage(content=query)],
            "query": query,
            "rag_results": [],
            "classification": {},
            "risk_assessment": {},
            "pattern_analysis": {},
            "compliance_check": {},
            "immediate_actions": [],
            "long_term_recommendations": [],
            "training_needs": [],
            "hazard_alerts": [],
            "current_agent": "",
            "agent_outputs": {}
        }
        
        try:
            final_state = self.graph.invoke(initial_state)
            
            return {
                "query": query,
                "classification": final_state["classification"],
                "risk_analysis": final_state["pattern_analysis"],
                "compliance": final_state["compliance_check"],
                "training": final_state.get("agent_outputs", {}).get("training", {}),
                "alerts": final_state["hazard_alerts"],
                "immediate_actions": final_state["immediate_actions"],
                "long_term_recommendations": final_state["long_term_recommendations"],
                "source_documents": len(final_state["rag_results"]),
                "agents_consulted": list(final_state["agent_outputs"].keys())
            }
        except Exception as e:
            return {
                "query": query,
                "error": str(e),
                "classification": {},
                "alerts": ["Error in agent processing"],
                "immediate_actions": [],
                "long_term_recommendations": []
            }


def create_safety_agent(rag_system) -> MultiAgentSafetySystem:
    """Factory function to create multi-agent safety system"""
    return MultiAgentSafetySystem(rag_system)