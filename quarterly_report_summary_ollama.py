#!/usr/bin/env python3
"""
Invoke Ollama gemma3:1b to distill quarterly performance reports into executive summaries.
Fetches report URL content, then uses system + human prompt for analysis.
"""

import re
from html.parser import HTMLParser
from urllib.request import Request, urlopen
from urllib.error import URLError

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate


class _TextExtractor(HTMLParser):
    def __init__(self):
        super().__init__()
        self.text_parts = []

    def handle_data(self, data):
        self.text_parts.append(data)

    def get_text(self):
        return " ".join(self.text_parts)


def fetch_url_content(url: str, max_chars: int = 80_000) -> str:
    """Fetch URL and return plain text (strip HTML). Truncate to max_chars for context limits."""
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; summary-bot/1.0)"})
    with urlopen(req, timeout=30) as resp:
        raw = resp.read().decode(errors="replace")
    parser = _TextExtractor()
    parser.feed(raw)
    text = re.sub(r"\s+", " ", parser.get_text()).strip()
    return text[:max_chars] if len(text) > max_chars else text

SYSTEM_PROMPT = """You are an expert Chief of Staff and Financial Analyst. Your task is to distill lengthy quarterly performance reports into high-density executive summaries for C-suite leaders.

**Strict Operational Rules:**
1. **Fact-Only Mandate:** Only include data points, metrics, and risks explicitly stated in the provided text. If a metric is missing (e.g., "Revenue is up" but no % is given), report it as "Revenue: [Growth cited; specific % not provided]".
2. **Role-Based Perspective:** Analyze the data from the perspective of a CEO/CFO—focus on bottom-line impact, operational efficiency, and strategic risk.
3. **No Hallucinations:** Do not infer "future success" or "strong momentum" unless the report explicitly uses those terms or provides the supporting data.
4. **Conciseness:** Keep the narrative summary under 150 words. Use active voice and eliminate filler phrases like "It is important to note that."  """

HUMAN_PROMPT = """Please analyze the attached Quarterly Performance Report and provide a concise executive summary.

**Report URL:** {report_url}

**Report content (fetched from URL):**
---
{report_content}
---

**Required Output Format:**
1. **Executive Summary:** A high-level narrative of the quarter's performance (Maximum 150 words).
2. **Key Metrics Table:** A Markdown table with three columns: [Metric Name, Current Value, Quarter-over-Quarter Change].
3. **Risks & Opportunities:** A bulleted section identifying:
    - Top 3 Operational or Financial Risks.
    - Top 2 Strategic Opportunities identified in the report.

**Constraint:** Ensure all metrics and risks are cited/sourced from the text provided. If the text does not contain enough data for a full table, provide only the confirmed data points."""

llm = ChatOllama(model="gemma3:1b", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])

chain = prompt | llm

report_url = "https://www.microsoft.com/en-us/investor/earnings/fy-2025-q4/press-release-webcast"

print("Fetching report content...")
try:
    report_content = fetch_url_content(report_url)
except URLError as e:
    report_content = f"[Could not fetch URL: {e}. Summarize based on the URL only if possible.]"
    print(f"Warning: {e}")

response = chain.invoke({"report_url": report_url, "report_content": report_content})
print(response.content)
