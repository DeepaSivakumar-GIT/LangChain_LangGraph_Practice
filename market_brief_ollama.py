#!/usr/bin/env python3
"""
Invoke Ollama Gemma 3b to generate a Market Analysis Brief.
Uses system prompt (Senior Market Intelligence Analyst) + human prompt with source links.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a Senior Market Intelligence Analyst. Your role is to synthesize complex, multi-source data into a high-level strategic brief.

**Core Directives:**
1. **Source Fidelity:** Base all analysis strictly on provided articles and reports. If information is missing for a SWOT quadrant, state "Insufficient data in sources" rather than speculating.
2. **Citation Standard:** Every claim, trend, and SWOT point must include an in-text citation (e.g., [Article 1] or [Report A]).
3. **Structured Output:** You must provide a dual-format response:
   - A valid JSON object for automated data ingestion.
   - A narrative executive summary for human stakeholders.
4. **Trend Identification:** Isolate the top 3 market drivers. Each must be supported by at least two distinct source references.
5. **SWOT Logic:**
   - Strengths/Weaknesses: Internal to the entity/market focus.
   - Opportunities/Threats: External macro-environmental factors.

**Output Schema Requirements:**
The JSON object must follow this key structure:
{{
  "market_brief": {{
    "top_trends": [{{"trend": "", "impact": "", "citations": []}}],
    "swot_analysis": {{"strengths": [], "weaknesses": [], "opportunities": [], "threats": []}}
  }}
}}"""

HUMAN_PROMPT = """Please generate a Market Analysis Brief based on the following sources:

**Sources:**
[SOURCE 1: {source_1}]
[SOURCE 2: {source_2}]
[SOURCE 3: {source_3}]

**Requirements:**
1. **JSON Data:** Create a structured object containing the SWOT analysis and the top 3 trends.
2. **Narrative Summary:** Write a 300-word executive summary that weaves the SWOT findings into a cohesive market outlook.
3. **Trend Focus:** Identify the 3 most significant trends and explain their implications for the next 12 months.
4. **Guardrail:** Ensure every trend and SWOT point has a corresponding citation from the sources provided above.

**Format:**
Deliver the JSON object first, followed by a horizontal rule, and then the Narrative Summary."""


llm = ChatOllama(model="gemma3:1b", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])

chain = prompt | llm

# Replace with your actual article URLs, report links, or pasted text excerpts
sources = {
    "source_1": "https://www2.deloitte.com/us/en/pages/technology/articles/2025-technology-industry-outlook.html",
    "source_2": "https://www.wsj.com/articles/us-defense-department-ai-llm-federal-budget-11675418000",
    "source_3": "https://www.theregister.com/2026/02/03/ai_llm_us_federal_budget/",
}

response = chain.invoke(sources)
print(response.content)
