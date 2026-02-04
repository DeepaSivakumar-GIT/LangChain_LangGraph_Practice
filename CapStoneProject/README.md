# Customer Support Email Agent

An AI agent that automatically processes incoming customer support emails—classifying urgency and topic, searching a knowledge base, drafting responses, and escalating complex cases to human agents.

Built with **LangChain** and **LangGraph** using the local **Ollama** model `gemma3:1b`.

## Features

- **Email classification**: Urgency (Low, Medium, High) and Topic (Account, Billing, Bug, Feature Request, Technical Issue)
- **Knowledge base search**: Retrieves relevant documentation and FAQs
- **Response drafting**: Generates suitable email replies using retrieved context
- **Escalation logic**: Routes complex or unresolved issues to human support
- **Follow-up scheduling**: Suggests follow-up actions when needed

## Requirements

- Python 3.10+
- [Ollama](https://ollama.ai/) installed with `gemma3:1b`:
  ```bash
  ollama pull gemma3:1b
  ollama serve  # if not already running
  ```

## Setup

```bash
cd customer-support-agent
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

**Process a single email (command line):**
```bash
python main.py "How do I reset my password?"
```

**Interactive mode (paste email, then Ctrl-D / Ctrl-Z):**
```bash
python main.py
```

**JSON output:**
```bash
python main.py "The export feature crashes when I select PDF format." --json
```

**Run all 5 example scenarios:**
```bash
python run_examples.py
```

**View the LangGraph workflow:**
```bash
python view_graph.py
```
Creates `graph.png` and prints Mermaid syntax (paste into [mermaid.live](https://mermaid.live) to visualize).

## Example Scenarios

| Scenario | Expected Behavior |
|----------|-------------------|
| Simple product question: "How do I reset my password?" | Auto-reply with KB steps |
| Bug report: "Export crashes when I select PDF" | Draft + often escalate for engineering |
| Urgent billing: "I was charged twice!" | Escalate, follow-up for refund |
| Feature request: "Add dark mode to mobile app" | Auto-reply, acknowledge request |
| Complex technical: "API fails intermittently with 504" | Escalate to engineering |

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌───────────────┐    ┌───────────────┐
│  classify   │───▶│  search_kb   │───▶│ draft_response│───▶│ decide_action │
│ (urgency,   │    │ (KB lookup)  │    │ (LLM draft)   │    │ (escalate?    │
│  topic)     │    │              │    │               │    │  follow-up?)  │
└─────────────┘    └──────────────┘    └───────────────┘    └───────────────┘
```

- **classify**: LLM classifies urgency and topic
- **search_kb**: In-memory knowledge base retrieval
- **draft_response**: LLM drafts reply using KB context
- **decide_action**: LLM + rules for escalate vs auto-reply and follow-ups

## Output

1. **Classified urgency** (Low / Medium / High)
2. **Identified topic** (Account, Billing, Bug, Feature Request, Technical Issue)
3. **Generated response draft**
4. **Decision** (Auto-reply vs Escalate)
5. **Follow-up action** (if any)

## Project Structure

```
customer-support-agent/
├── main.py              # Entry point
├── run_examples.py      # Run all 5 example scenarios
├── view_graph.py        # View LangGraph workflow (graph.png + Mermaid)
├── requirements.txt
├── README.md
└── src/
    ├── __init__.py
    ├── agent.py         # LangGraph workflow
    └── knowledge_base.py # FAQ/documentation
```

## Extending

- **Knowledge base**: Edit `src/knowledge_base.py` or connect a vector store
- **Escalation rules**: Adjust logic in `decide_action` in `src/agent.py`
- **Model**: Change `model="gemma3:1b"` in `src/agent.py` for a different Ollama model

## License

MIT
