"""
Module 12 - Agent Architecture
Lesson 03 - Multiple Tools, One Bind

Goal
----
- Define the remaining three on-call tools
- Bind all four at once and watch the model pick the right one per question
"""

from langchain_core.tools import tool
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector


# -------------------------------------------------
# PostgreSQL Connection
# -------------------------------------------------

CONNECTION = (
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)

vector_store = PGVector(
    embeddings=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3,
    }
)


# -------------------------------------------------
# Tool: Search Knowledge Base
# -------------------------------------------------

@tool
def search_knowledge_base(
    query: str,
) -> str:
    """Search the internal knowledge base for troubleshooting
    articles and past ticket resolutions. Use this when the
    user's question could be answered from documented procedures."""

    documents = retriever.invoke(query)

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# -------------------------------------------------
# Tool: Get Ticket Status
# -------------------------------------------------

@tool
def get_ticket_status(
    ticket_id: str,
) -> str:
    """Look up the current status of a support ticket by its ID.
    Returns state, priority, and last update time."""

    fake_tickets = {
        "T-1001": "OPEN, priority=P2, last_updated=2h ago",
        "T-1002": "RESOLVED, priority=P3, last_updated=1d ago",
    }

    return fake_tickets.get(ticket_id, "Ticket ID not found.")


# -------------------------------------------------
# Tool: Check Service Health
# -------------------------------------------------

@tool
def check_service_health(
    service_name: str,
) -> str:
    """Check whether a named backend service is currently healthy,
    degraded, or down. Use before escalating an infra-related ticket."""

    fake_health = {
        "billing-api": "DEGRADED - elevated latency since 09:00 UTC",
        "auth-service": "HEALTHY",
    }

    return fake_health.get(service_name, "Unknown service.")


# -------------------------------------------------
# Tool: Compute SLA Breach
# -------------------------------------------------

@tool
def compute_sla_breach(
    hours_open: float,
    priority: str,
) -> str:
    """Determine whether a ticket has breached its SLA window
    given how many hours it has been open and its priority level."""

    sla_hours = {
        "P1": 4,
        "P2": 24,
        "P3": 72,
    }

    limit = sla_hours.get(priority.upper(), 72)

    if hours_open > limit:
        return f"SLA BREACHED - open {hours_open}h, limit {limit}h"

    return f"Within SLA - open {hours_open}h, limit {limit}h"


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# -------------------------------------------------
# Bind All Four Tools
# -------------------------------------------------

tools = [
    search_knowledge_base,
    get_ticket_status,
    check_service_health,
    compute_sla_breach,
]

llm_with_tools = llm.bind_tools(tools)


# -------------------------------------------------
# Ask Three Different Questions - Each Should Trigger A Different Tool
# -------------------------------------------------

test_questions = [
    "What's the status of ticket T-1001?",
    "Is billing-api healthy right now?",
    "A ticket has been open for 30 hours at P2 priority, has it breached SLA?",
]

print("\n" + "=" * 60)
print("MULTI-TOOL SELECTION TEST")
print("=" * 60)

for question in test_questions:

    response = llm_with_tools.invoke(question)

    print(f"\nQuestion: {question}")
    print(f"Tool Calls: {response.tool_calls}")
