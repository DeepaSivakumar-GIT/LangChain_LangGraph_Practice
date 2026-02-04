"""
Customer Support Email Agent - LangGraph workflow.

Processes incoming emails: classify → search KB → draft response → decide action.
"""

from typing import TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from .knowledge_base import search_knowledge_base

# --- State schema ---


class EmailState(TypedDict):
    """State passed through the email processing graph."""

    email_content: str
    urgency: str
    topic: str
    kb_context: str
    response_draft: str
    escalate: bool
    follow_up: str


# --- LLM setup ---

LLM = ChatOllama(model="gemma3:1b", temperature=0.2)

# --- Classification prompt ---

CLASSIFY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a customer support classifier. Classify the email exactly.\n"
        "Urgency: Low (general questions), Medium (needs help soon), High (urgent, e.g. billing errors, outages)\n"
        "Topic: Account (password, login, profile), Billing (charges, subscription), "
        "Bug (crashes, errors), Feature Request (new feature), Technical Issue (API, integration)\n"
        "Respond with exactly: URGENCY|TOPIC (e.g., Low|Account or High|Billing)",
    ),
    ("human", "{email}"),
])

# --- Response draft prompt ---

DRAFT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a professional customer support agent. Draft a helpful, empathetic email response. "
        "Use the provided knowledge base context when relevant. Be concise. Do not invent information.\n\n"
        "Knowledge base:\n{kb_context}\n\n"
        "If the KB does not fully address the issue, acknowledge it and suggest next steps.",
    ),
    ("human", "Customer email:\n{email}\n\nDraft a response."),
])

# --- Decision prompt ---

DECIDE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        "Decide whether to AUTO_REPLY or ESCALATE based on:\n"
        "- Urgency High + Billing/Technical Issue → ESCALATE\n"
        "- Bug reports needing engineering → ESCALATE\n"
        "- Complex Technical Issue (e.g., intermittent API errors) → ESCALATE\n"
        "- Simple questions, feature requests → AUTO_REPLY\n"
        "If escalating, suggest a follow-up action (e.g., 'Human to review within 24h').\n"
        "Respond with: AUTO_REPLY or ESCALATE\n"
        "Then on a new line: follow-up action or 'None'",
    ),
    (
        "human",
        "Email topic: {topic}, Urgency: {urgency}\nEmail: {email}\nDraft: {draft}\n\nDecision:",
    ),
])


# --- Graph nodes ---


def classify_email(state: EmailState) -> dict:
    """Classify email by urgency and topic."""
    chain = CLASSIFY_PROMPT | LLM
    result = chain.invoke({"email": state["email_content"]})
    text = result.content.strip()
    parts = text.split("|")
    urgency = parts[0].strip() if len(parts) > 0 else "Medium"
    topic = parts[1].strip() if len(parts) > 1 else "Technical Issue"

    # Normalize
    if urgency not in ("Low", "Medium", "High"):
        urgency = "Medium"
    valid_topics = ("Account", "Billing", "Bug", "Feature Request", "Technical Issue")
    if topic not in valid_topics:
        topic = "Technical Issue"

    # Keyword fallback for common misclassifications (small models)
    email_lower = state["email_content"].lower()
    if "password" in email_lower or "reset" in email_lower or "login" in email_lower:
        if "api" not in email_lower and "504" not in email_lower:
            topic = "Account"
    if "charged" in email_lower or "double charge" in email_lower or "billing" in email_lower:
        topic = "Billing"
    if "crash" in email_lower or "bug" in email_lower or ("export" in email_lower and "pdf" in email_lower):
        topic = "Bug"
    if "dark mode" in email_lower or "feature" in email_lower and "add" in email_lower:
        topic = "Feature Request"

    return {"urgency": urgency, "topic": topic}


def search_kb(state: EmailState) -> dict:
    """Search knowledge base for relevant content."""
    context = search_knowledge_base(state["email_content"], state["topic"])
    return {"kb_context": context}


def draft_response(state: EmailState) -> dict:
    """Draft customer response using KB context."""
    chain = DRAFT_PROMPT | LLM
    result = chain.invoke({
        "email": state["email_content"],
        "kb_context": state["kb_context"],
    })
    return {"response_draft": result.content.strip()}


def decide_action(state: EmailState) -> dict:
    """Decide: auto-reply vs escalate, and any follow-up."""
    chain = DECIDE_PROMPT | LLM
    result = chain.invoke({
        "email": state["email_content"],
        "topic": state["topic"],
        "urgency": state["urgency"],
        "draft": state["response_draft"],
    })
    text = result.content.strip().upper()
    escalate = "ESCALATE" in text
    lines = result.content.strip().split("\n")
    follow_up = lines[-1].strip() if len(lines) > 1 and "none" not in lines[-1].lower() else "None"

    # Rule-based overrides for robustness
    if state["urgency"] == "High" and state["topic"] in ("Billing", "Technical Issue"):
        escalate = True
    if "504" in state["email_content"] or "intermittent" in state["email_content"].lower():
        escalate = True
        follow_up = "Engineering to investigate API errors within 48h"
    # Simple Account questions (password reset, etc.) → auto-reply
    if state["topic"] == "Account" and state["urgency"] != "High":
        escalate = False

    return {"escalate": escalate, "follow_up": follow_up if follow_up != "None" else ""}


# --- Build graph ---


def build_graph() -> CompiledStateGraph:
    """Build and compile the customer support email graph."""
    builder = StateGraph(EmailState)

    builder.add_node("classify", classify_email)
    builder.add_node("search_kb", search_kb)
    builder.add_node("draft_response", draft_response)
    builder.add_node("decide_action", decide_action)

    builder.add_edge(START, "classify")
    builder.add_edge("classify", "search_kb")
    builder.add_edge("search_kb", "draft_response")
    builder.add_edge("draft_response", "decide_action")
    builder.add_edge("decide_action", END)

    return builder.compile()


# Singleton graph instance
_graph: CompiledStateGraph | None = None


def get_agent() -> CompiledStateGraph:
    """Get the compiled email processing agent."""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph
