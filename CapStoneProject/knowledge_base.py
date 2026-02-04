"""Simple in-memory knowledge base for customer support FAQs."""

KNOWLEDGE_BASE = {
    "account": [
        "To reset your password: 1) Go to Settings > Account, 2) Click 'Forgot Password', "
        "3) Enter your email to receive a reset link. Link expires in 1 hour.",
        "Account recovery: Use the 'Forgot Username' link on the login page. "
        "You will need to verify your email address.",
    ],
    "billing": [
        "For duplicate charges: Contact support with your transaction ID. "
        "Refunds are processed within 5-7 business days.",
        "Subscription management: Go to Settings > Billing to cancel or change your plan. "
        "Changes take effect at the next billing cycle.",
    ],
    "bug": [
        "Known issue - PDF export: We are aware of crashes when selecting PDF format. "
        "Workaround: Try exporting as CSV first, then convert. Fix planned for next release.",
        "Bug reports: Please include: steps to reproduce, your app version, and device/browser. "
        "Submit via our support portal for tracking.",
    ],
    "feature_request": [
        "We log all feature requests. Dark mode, mobile improvements, and API enhancements "
        "are on our roadmap. Vote for features in our community forum.",
        "Feature request process: Submit via the Feedback form. Our product team reviews "
        "requests quarterly.",
    ],
    "technical_issue": [
        "API 504 errors: Usually indicate timeout or temporary server overload. "
        "Implement retry with exponential backoff. Check status page for outages.",
        "API integration: Ensure you use the latest API version. Rate limit: 100 req/min. "
        "Documentation: docs.example.com/api",
    ],
}


def search_knowledge_base(query: str, topic: str, k: int = 3) -> str:
    """Search the knowledge base for relevant content."""
    results = []
    query_lower = query.lower()
    topic_lower = topic.lower().replace(" ", "_").replace("featurerequest", "feature_request")

    # Map topic to KB keys
    topic_key = topic_lower if topic_lower in KNOWLEDGE_BASE else "technical_issue"
    if topic_key in KNOWLEDGE_BASE:
        results.extend(KNOWLEDGE_BASE[topic_key])

    # Also search by query keywords across all sections
    for section, docs in KNOWLEDGE_BASE.items():
        if section != topic_key:
            for doc in docs:
                if any(word in doc.lower() for word in query_lower.split()):
                    if doc not in results:
                        results.append(doc)

    return "\n\n".join(results[:k]) if results else "No specific documentation found. Suggest escalation for complex queries."
