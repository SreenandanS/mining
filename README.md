# SARSA : AI for Mine Safety Intelligence

### A Digital Mining Safety Officer powered by Agentic AI + Real-time RAG

**SARSA (Safety & Risk Smart Assistant)** is one of the fastest (Pathway framework) agent-powered mine-safety intelligence platform designed for real-time accident monitoring, analysis, and autonomous safety auditing.


---

## 🚀 Features

SARSA provides an end-to-end autonomous system for monitoring, querying, and auditing mine safety data.

| Module | Function |
| :--- | :--- |
| **Analysis Agent** | Computes KPIs, heatmaps, timelines, and hazard trends. |
| **Query Agent** | Provides natural-language Q&A for mine safety data. |
| **Audit Agent** | Auto-generates DGMS-aligned safety audit reports. |
| **Regulatory RAG** | Reasons over compliance with The Mines Act & MMR 1961. |
| **Live FS/news Data Stream**| Watches `data/dgms/` folder and parses new PDFs/news jsonls live. |
| **Embeddings / RAG** | Creates a semantic vector index of accidents and rules. |
| **LangGraph Orchestration**| Manages the multi-agent flow (Analysis → Query → Audit). |
| **Streamlit Frontend** | Powers the dashboards, chat interface, and audit console. |
| **Live Alerts** | New accidents auto-appear with toast notifications. |

## 📂 System Pipeline

The flow of data from raw PDF reports to actionable insights in the UI.


DGMS PDFs / Live Folder
        ↓
PyMuPDF + custom parser
        ↓
Structured accident dataset (date, mine, location, cause, victims, description)
        ↓
Sentence-Transformer embeddings + FAISS (Vector Index)
        ↓
LangGraph Agents:
    • Analysis Agent
    • Query Agent
    • Audit Agent
        ↓
RAG over regulatory knowledge base (MMR 1961, Mines Act)
        ↓
Streamlit UI + Live Pathway Streaming + OpenAI conversational layer
1️⃣ Analysis Agent

Fatality trends

Hazard category stats

State-wise patterns

Heatmaps + timelines

2️⃣ Query Agent

Free-form natural-language queries:

“Show dumper-related incidents in Jharkhand mines in monsoon months.”
“Summarize ladder-fall hazards in UG coal mines.”

3️⃣ Audit Agent

Maps incidents to The Mines Act & MMR rules

Flags probable violations

Generates preventive recommendations

📘 Regulatory Knowledge Base

Examples:

Incident	Rule Check
Fall from height	MMR 118(4) — safety belt requirement
Gas explosion	MMR 124 — ventilation/gas monitoring
## ⚡ Real-time Pipeline
🗂️ Live PDF/ews streaming Kafka/Drop-Folder (Local FS)

Drop DGMS reports into:

data/dgms/

## 📡 Pathway FS Streamer (continuous ingestion)

Runs separately:
python pathway_ingestor_fs.py
Watches folder → parses PDF → writes cleaned rows → out/jsonl/

## 📊 Streamlit UI auto-updates

New incidents appear in Live Events panel
FAISS index refreshes only on new rows
Toast alerts for new events

## 🏗️ Tech Stack
Layer	Technology
Agents & Orchestration:	LangGraph
LLM: OpenAI GPT-5 / GPT-4o-mini
NLP/RAG:	Sentence Transformers + FAISS
Streaming:	Pathway filesystem streaming
Frontend:	Streamlit
PDF parsing:	PyMuPDF + **custom rule-based extraction**
Data:	DGMS historical accident reports
Reg Compliance	MMR 1961 + The Mines Act 1952
📦 Installation
git clone <repo>
cd sarsa
pip install -r requirements.txt

Environment Variables

Create .env

OPENAI_API_KEY="sk-xxxx"

▶️ Run System
1️⃣ Start FS Streaming (live ingestion)
python pathway_ingestor_fs.py


This watches data/dgms/ and pipes parsed accidents to out/jsonl/

2️⃣ Start UI
streamlit run app.py

🖥️ App Screens
📊 Executive Dashboard

KPIs (fatalities, accidents, hazard types)
Heatmap
Accident timeline
Live Events feed

💬 Interactive Query
Conversational answers
Token streaming UX
Structured JSON view for transparency

📝 Safety Audit Generator

Auto-audit based on accident logs
Regulatory rule justification
Safety recommendations

🎯 Capabilities Demonstrated in PPT
Concept	Implemented
Agentic AI (3 agents)	✅
RAG + KG like compliance memory	✅
Real-time ingestion	✅ (local folder)
Predictive trends:	Partial 
Regulatory rule mapping	✅ MMR/Mines Act layer
Live dashboard	✅

## 🔮 Future Scope

Real IoT sensor ingestion (vibration, gas monitors)
Hindi/Multilingual mining safety chatbot
DGMS cloud deployment
Safety video analytics integration
Risk score per mine (seasonality / geo-risk)

## 🧪 Example Test Flow

Place DGMS PDFs/news stream into ./data/dgms/
Start streamer → UI picks records live
Ask queries like:
“Accidents involving HEMM in Chhattisgarh last 2 years”
Generate safety audit PDF
Live-refresh to see auto-alerts

## 👥 Team

IIT(ISM) Dhanbad
Sreenandan Shashidharan, Anukul Tiwari, Raj Priyadarshi, Suryansh Kulshreshth, Ayushman Dutta

## 🛡️ Safety Commitment

SARSA aims to save lives in mines through intelligence, insight, and autonomy — reducing accidents and strengthening compliance.
