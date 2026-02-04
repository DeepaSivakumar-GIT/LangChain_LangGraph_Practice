#!/usr/bin/env python3
"""
Invoke Ollama gemma3:1b to audit a draft HR policy.
Uses system prompt (Senior HR Compliance Auditor) + human prompt with draft policy and region.
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

SYSTEM_PROMPT = """You are a Senior HR Compliance Auditor. Your role is to review draft policy documents for legal safety, clarity, and completeness.

**Operational Guidelines:**
1. **Gap Analysis:** Identify missing standard clauses (e.g., EEO statements, At-Will disclaimers, Privacy notices) based on general labor law standards and the provided context.
2. **Ambiguity Detection:** Flag phrases like "at the manager's discretion," "often," or "as soon as possible" as high-risk due to lack of specificity.
3. **Severity Rating:** - **Critical:** Legal liability or regulatory non-compliance.
   - **Moderate:** Operational confusion or potential for employee disputes.
   - **Low:** Minor grammatical or stylistic improvements.
4. **Citations:** When flagging an issue, cite the specific section of the draft policy.
5. **JSON Constraint:** You must output valid JSON. Do not include prose outside of the JSON block.

**JSON Schema:**
{{
  "Issues": [
    {{
      "id": 1,
      "policy_reference": "Section/Paragraph string",
      "issue_type": "Missing Clause | Ambiguity | Improvement",
      "description": "Detailed explanation of the problem",
      "severity": "Critical | Moderate | Low",
      "recommendations": "Specific wording or clause to insert"
    }}
  ]
}}"""

HUMAN_PROMPT = """Please perform a compliance and clarity audit on the following draft HR policy.

**Draft Policy Text:**
{draft_policy}

**Requirements:**
1. Identify missing essential clauses for a {region} based company.
2. Flag ambiguous language that could lead to inconsistent enforcement.
3. Suggest specific improvements to make the text more professional and legally robust.

**Output Format:**
Return only a JSON object following the schema provided in your system instructions."""

DEFAULT_DRAFT_POLICY = """Remote Work and Equipment Policy: Global Connectivity & Remote Work Guidelines
1. Introduction This document outlines the expectations for employees working from home. We want to be a flexible workplace, so we allow people to work remotely when it makes sense for their roles.
2. Eligibility Remote work is generally available to most office-based employees. Approval is typically handled by your direct manager. They will decide if your performance is good enough to warrant working from home. We expect you to be online during "normal" business hours, though some flexibility is allowed if you have errands to run.
3. Equipment & Expenses The company may provide a laptop for work use. Employees are responsible for their own internet connection. If you need a monitor or a chair, you can ask your manager, and they might approve a reimbursement if there is room in the budget this quarter. Please don't break the equipment; if you do, we might ask you to pay for it.
4. Behavior & Productivity We trust our employees to be productive. You don't need to log your hours specifically as long as your work gets done. However, if we notice you aren't responding to Slack messages quickly, we may revoke your remote work privileges at any time without much notice.
5. Safety Please make sure your home office is safe and ergonomic. The company is not responsible for any accidents that happen while you are working in your living room or a coffee shop."""

llm = ChatOllama(model="gemma3:1b", temperature=0.2)

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("human", HUMAN_PROMPT),
])

chain = prompt | llm

inputs = {
    "draft_policy": DEFAULT_DRAFT_POLICY,
    "region": "US/California",
}

response = chain.invoke(inputs)
print(response.content)
