# AI for Proactive Mine Safety Intelligence

## Overview

Mining accidents have historically been a significant concern in India. This project aims to address this challenge by developing an intelligent platform designed to provide precautions, deliver timely warnings, and suggest actionable safety measures to prevent incidents.

The core goal is to shift from reactive analysis to proactive prevention. The system will leverage Artificial Intelligence to analyze historical accident data and monitor real-time information streams—including official reports and local news—to identify emerging hazards before they result in an accident.

## Project Aim

This project aims to use the potential of NLP to digitize and analyze extensive collections of Indian mining accident records. The platform will analyze data to detect patterns, and its advanced AI agents will autonomously monitor new information to provide warnings and recommend specific preventive measures.

## Getting Started

Follow these steps to set up the project locally.

### 1. Clone the Repository

```bash
git clone [https://github.com/your-username/your-repository-name.git](https://github.com/your-username/your-repository-name.git)
cd your-repository-name
```

### 2. make a new python environment

```bash
# Create a new environment named 'venv'
python -m venv venv
# Activate the environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. install dependecies

```bash
pip install -r requirements.txt
```

## Key Features

### 1. Core Data Analysis

* **Pattern Detection:** The system will use AI techniques to allow users to quickly access insights and detect patterns in accident data.
* **Interactive Dashboard:** Users can view real-time accident trends, locations, and timelines through a simple, interactive interface.
* **Hazard Identification:** The platform is designed to outperform traditional methods in identifying hazards and investigating root causes.
* **Automated Reporting:** It will generate automated safety audit reports, which reduces human labor and increases accuracy.

### 2. Proactive "Agentic AI" Capabilities

This project moves beyond simple analysis by incorporating autonomous agents that act on data:

* **Autonomous Safety Monitoring Agents**
   * **Proactive Alerts:** The system will automatically classify incidents, flag potential hazards, and generate alerts (e.g., “Increase in transportation machinery accidents in Jharkhand mines in Q3 2022”).
    * **Incident Analysis & Compliance:** When a new accident is detected (e.g., from news data), the AI will analyze the event to determine which **code of conduct or rule was not followed** and suggest specific **steps to prevent it**.
    * **Preventive Recommendations:** Based on the data, agents will recommend targeted inspections or specific preventive measures to mine operators.
* **Interactive “Digital Mine Safety Officer”**
    * **Conversational AI:** A conversational layer built on the NLP platform will allow users to ask domain-specific questions (e.g., “Show me all methane-related accidents in 2021 in underground coal mines”).
    * **Compliance Suggestions:** The agent can suggest regulatory compliance actions (e.g., “Mine X exceeds threshold for ground movement incidents; schedule slope stability inspection”).

## Data Source

The data for this analysis originates from the Directorate General of Mines Safety (DGMS), India. The study utilizes more than 300 incident records from the years 2016 to 2022.
