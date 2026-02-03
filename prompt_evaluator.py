#!/usr/bin/env python3
"""
Simple prompt quality evaluator (Ollama, default model, formatted output only).

Usage:
  python prompt_evaluator.py "Your prompt here"
  python prompt_evaluator.py   # then type prompt, Ctrl-D / Ctrl-Z to finish

Requires: Ollama running locally, pip install langchain-core langchain-ollama pydantic
"""

import sys

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import BaseModel, Field


# --- Output schema ---
class CriterionScores(BaseModel):
    clarity: int = Field(ge=0, le=10, description="Clarity: easy to understand, clear goal")
    specificity_details: int = Field(ge=0, le=10, description="Specificity: sufficient details")
    context: int = Field(ge=0, le=10, description="Context: background, audience, use case")
    output_format_constraints: int = Field(ge=0, le=10, description="Format: tone, length, format")
    persona_defined: int = Field(ge=0, le=10, description="Persona: specific role for AI")


class PromptEvaluationResult(BaseModel):
    criterion_scores: CriterionScores
    explanation: str
    improvement_suggestions: list[str]


# --- Prompts ---
SYSTEM_PROMPT = """You are an expert prompt engineer. Evaluate user prompts for AI systems.

Score 0-10 on each criterion:
1. Clarity: Easy to understand, clear goal?
2. Specificity/Details: Enough detail and requirements?
3. Context: Background, audience, or use case?
4. Output Format & Constraints: Format, tone, length specified?
5. Persona Defined: Specific role assigned to the AI?

Be objective. Give 2-3 actionable improvement suggestions."""

USER_TEMPLATE = """Evaluate this prompt:

---
{prompt}
---"""


def evaluate(prompt: str, temperature: float = 0.2) -> dict:
    """Run evaluation (single chain, default model, no fallback)."""
    llm = ChatOllama(model="gemma3:1b", temperature=temperature)
    structured_llm = llm.with_structured_output(PromptEvaluationResult)
    chat_prompt = ChatPromptTemplate.from_messages([
        ("system", SYSTEM_PROMPT),
        ("human", USER_TEMPLATE),
    ])
    chain = chat_prompt | structured_llm

    result: PromptEvaluationResult = chain.invoke({"prompt": prompt})

    s = result.criterion_scores
    final = (s.clarity + s.specificity_details + s.context + s.output_format_constraints + s.persona_defined) / 5.0

    return {
        "final_score": round(final, 1),
        "criterion_scores": {
            "clarity": s.clarity,
            "specificity_details": s.specificity_details,
            "context": s.context,
            "output_format_constraints": s.output_format_constraints,
            "persona_defined": s.persona_defined,
        },
        "explanation": result.explanation,
        "improvement_suggestions": result.improvement_suggestions,
    }


def print_result(result: dict) -> None:
    print("\n" + "=" * 60)
    print("PROMPT QUALITY SCORE")
    print("=" * 60)
    print(f"\nFinal Score: {result['final_score']}/10\n")
    print("Criterion Scores:")
    for name, score in result["criterion_scores"].items():
        print(f"  • {name.replace('_', ' ').title()}: {score}/10")
    print(f"\nExplanation:\n  {result['explanation']}")
    print("\nImprovement Suggestions:")
    for i, s in enumerate(result["improvement_suggestions"], 1):
        print(f"  {i}. {s}")
    print("=" * 60 + "\n")


def main() -> None:
    if len(sys.argv) > 1:
        text = sys.argv[1]
    else:
        print("Enter prompt (Ctrl-D / Ctrl-Z to finish):", flush=True)
        try:
            text = sys.stdin.read().strip()
        except EOFError:
            print("No input.", file=sys.stderr)
            sys.exit(1)

    if not text:
        print("Empty prompt.", file=sys.stderr)
        sys.exit(1)

    try:
        result = evaluate(text)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_result(result)


if __name__ == "__main__":
    main()
