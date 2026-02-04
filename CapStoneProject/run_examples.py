#!/usr/bin/env python3
"""
Run all 5 example scenarios from the capstone requirements.
"""

from main import format_output
from src.agent import get_agent

EXAMPLES = [
#    "How do I reset my password?",
#    "The export feature crashes when I select PDF format.",
#    "I was charged twice for my subscription!",
#    "Can you add dark mode to the mobile app?",
    "Our API integration fails intermittently with 504 errors.",
]

LABELS = [
#    "1. Simple product question",
#    "2. Bug report",
#    "3. Urgent billing issue",
#    "4. Feature request",
    "5. Complex technical issue",
]


def main():
    agent = get_agent()
    for label, email in zip(LABELS, EXAMPLES):
        print(f"\n{'='*60}\n{label}\nEmail: {email}\n")
        initial = {
            "email_content": email,
            "urgency": "",
            "topic": "",
            "kb_context": "",
            "response_draft": "",
            "escalate": False,
            "follow_up": "",
        }
        result = agent.invoke(initial)
        print(format_output(result))


if __name__ == "__main__":
    main()
