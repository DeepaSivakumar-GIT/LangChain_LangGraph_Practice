#!/usr/bin/env python3
"""
Customer Support Email Agent - Main entry point.

Processes customer support emails using LangChain and LangGraph with Ollama (gemma3:1b).

Usage:
  python main.py "Your email content here"
  python main.py                    # Interactive mode
"""

import argparse
import json
import sys


def format_output(result: dict) -> str:
    """Format the agent output for display."""
    lines = [
        "=" * 50,
        "CUSTOMER SUPPORT EMAIL PROCESSING RESULT",
        "=" * 50,
        f"Classified Urgency: {result.get('urgency', 'N/A')}",
        f"Identified Topic:   {result.get('topic', 'N/A')}",
        "-" * 50,
        "Response Draft:",
        result.get("response_draft", "N/A"),
        "-" * 50,
        f"Decision:          {'ESCALATE to human' if result.get('escalate') else 'AUTO-REPLY'}",
        f"Follow-up Action:  {result.get('follow_up') or 'None'}",
        "=" * 50,
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process customer support emails with AI agent"
    )
    parser.add_argument(
        "email",
        nargs="?",
        default=None,
        help="Email content to process (or omit for interactive mode)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted text",
    )
    args = parser.parse_args()

    try:
        from src.agent import get_agent
    except ImportError as e:
        print(
            "Error: Install dependencies first:\n"
            "  source venv/bin/activate  # or venv\\Scripts\\activate on Windows\n"
            "  pip install -r requirements.txt",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.email:
        email_content = args.email
    else:
        print("Enter customer email (Ctrl-D / Ctrl-Z to finish):")
        print("-" * 40)
        try:
            email_content = sys.stdin.read().strip()
        except EOFError:
            print("No input provided.", file=sys.stderr)
            sys.exit(1)

    if not email_content:
        print("Error: Empty email content.", file=sys.stderr)
        sys.exit(1)

    # Build initial state
    initial_state = {
        "email_content": email_content,
        "urgency": "",
        "topic": "",
        "kb_context": "",
        "response_draft": "",
        "escalate": False,
        "follow_up": "",
    }

    agent = get_agent()
    result = agent.invoke(initial_state)

    if args.json:
        print(json.dumps(dict(result), indent=2))
    else:
        print(format_output(result))


if __name__ == "__main__":
    main()
