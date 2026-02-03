#!/usr/bin/env python3
"""
Beginner snippet: draft a project status email using Ollama Gemma 3b.
Uses a system prompt (Corporate Communications Assistant) + user prompt.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are an expert Corporate Communications Assistant. Your goal is to draft professional, high-clarity project update emails.

**Guidelines:**
1. Tone: Maintain a formal, proactive, and respectful tone at all times.
2. Structure: Every email must include a clear Subject Line, an introduction, a summary of progress, a section for milestones, and a clear list of Action Items.
3. Placeholders: Use [Deadline] as a placeholder where the feedback deadline is not yet set.
4. Guardrails: Do not invent specific financial figures or sensitive company data. Focus on process and delivery.
5. Clarity: Use bullet points for readability when listing tasks or milestones."""

USER_PROMPT = """Please draft a professional project status email based on the following details:

- Project Name: {project_name}
- Client: {client_name}
- Key Progress: We have completed the initial architectural audit and finalized the wireframes for the user dashboard.
- Milestones Reached: Phase 1 (Discovery) is 100% complete; Phase 2 (Design) is currently at 75%.
- Deadline for Feedback: [Deadline]
- Action Items:
    1. Client to review the wireframe PDF attached.
    2. Schedule a 15-minute sync for Thursday to approve design directions.
    3. Finalize the color palette selection.

Ensure the email invites the client to provide feedback and maintains a polished, executive-level feel."""

llm = ChatOllama(model="gemma3:1b", temperature=0.3)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
])

chain = prompt | llm

project_name = input("Project name: ").strip()
client_name = input("Client name: ").strip()

response = chain.invoke({"project_name": project_name, "client_name": client_name})

print(response.content)
