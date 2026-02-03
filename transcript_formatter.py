#!/usr/bin/env python3
"""
Beginner snippet: format meeting transcripts using Ollama gemma3:1b.
User picks one of two sample transcripts; the model extracts decisions and action items.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a highly precise Project Management Analyst. Your task is to extract actionable intelligence from raw meeting transcripts.

**Operational Rules:**
1. **Source Fidelity:** Only include decisions and action items explicitly stated or heavily implied in the transcript. If a detail (like an owner or deadline) is missing, mark it as [Unassigned] or [TBD].
2. **Confidence Scores:** For every item, provide a Confidence Score (0-100%). This reflects how clearly the transcript defines the item, its owner, and its deadline.
3. **Markdown Structure:** Use "## Decisions" and "## Action Items" as primary headers.
4. **Tone:** Neutral, professional, and concise. Use active voice for action items.
5. **Guardrail:** Do not invent data. If the transcript is ambiguous, reflect that ambiguity in the confidence score and description."""

USER_PROMPT = """Please analyze the following raw meeting transcript and organize it into a structured summary.

**Transcript:**
{transcript}

**Requirements:**
- Extract all key decisions made during the call.
- List all action items in a table or list format.
- For each action item, identify:
    - The Owner
    - The Deadline
    - A Confidence Score (based on the clarity of the recording/text)

**Format:**
## Decisions
- [Decision Point]

## Action Items
| Action Item | Owner | Deadline | Confidence Score |
| :--- | :--- | :--- | :--- |"""

SAMPLE_TRANSCRIPT_1 = """Transcript: Alex: Okay everyone, thanks for jumping on. We need to lock in the Q4 launch plan. First item: are we sticking with the November 15th release date for the mobile app? Jamie: Yes, the dev team confirmed that's doable, provided we freeze the feature set by this Friday. Alex: Great, so it's decided—November 15th is the hard launch. Jamie, I need you to handle that feature freeze memo and get it to the engineering leads by EOD tomorrow. Sarah: What about the promotional video? Alex: Sarah, that's on you. We need a final cut for the board meeting on October 20th. Can you manage that? Sarah: I'll have to check with the editor, but let's pencilled it in. I'll confirm by Wednesday. Alex: Also, we need someone to update the pricing page on the website. Jamie: I can take a look, but I don't have the login credentials yet. Alex: No worries, let's just get it done before the launch."""

SAMPLE_TRANSCRIPT_2 = """Mark: Total chaos on the server migration. Everything is lagging. Priya: I told the vendor we needed more bandwidth, but they haven't replied. Mark: Someone needs to call them. Like, right now. Priya, can you do that? Or maybe Dave? Dave: I'm tied up with the database fix, but I can call them if Priya is busy. Priya: No, I'll do it. I'll try to get them on the phone today. If not, I'll send an email. Mark: Okay, so Priya is calling the vendor. We also decided in the last meeting to stop the legacy backups, right? Dave: Yeah, we're killing the legacy backups effective immediately. That's a firm go. Mark: Good. Now, we need a post-mortem report on why the server crashed in the first place. Priya: Dave, you have the logs for that? Dave: I have some of them. I'll start drafting something. I'm not sure when it'll be ready though, probably sometime next week? Mark: Just get it to me as soon as possible. We can't have a repeat of this."""

SAMPLES = {
    "1": ("Q4 launch plan / mobile app", SAMPLE_TRANSCRIPT_1),
    "2": ("Server migration / vendor & post-mortem", SAMPLE_TRANSCRIPT_2),
}

llm = ChatOllama(model="gemma3:1b", temperature=0.2)
prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", USER_PROMPT),
])
chain = prompt | llm

print("Select a transcript to format:")
for key, (desc, _) in SAMPLES.items():
    print(f"  {key}. {desc}")
choice = input("Enter 1 or 2: ").strip()

if choice not in SAMPLES:
    print("Invalid choice. Using transcript 1.")
    choice = "1"

_, transcript = SAMPLES[choice]
print("\nFormatting transcript...\n")
response = chain.invoke({"transcript": transcript})
print(response.content)
