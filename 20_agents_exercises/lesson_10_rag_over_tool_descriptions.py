"""
Module 20 - Agent Architecture
Lesson 10 - RAG Over Tool Descriptions

Goal
----
- Simulate a large tool catalog (12+ tools)
- Embed tool descriptions into the complete_pipeline collection
- Retrieve only the relevant subset before binding to the LLM
"""

from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
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


# -------------------------------------------------
# Simulated Tools (Real Definitions, Not Just Names)
# -------------------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for troubleshooting articles."""
    return f"KB results for: {query}"


@tool
def get_ticket_status(ticket_id: str) -> str:
    """Look up the current status of a support ticket by its ID."""
    return f"Status for {ticket_id}"


@tool
def check_service_health(service_name: str) -> str:
    """Check whether a named backend service is healthy, degraded, or down."""
    return f"Health for {service_name}"


@tool
def compute_sla_breach(hours_open: float, priority: str) -> str:
    """Determine whether a ticket has breached its SLA window."""
    return "SLA check result"


@tool
def restart_service(service_name: str) -> str:
    """Restart a named backend service that is degraded or down."""
    return f"Restarted {service_name}"


@tool
def page_oncall_engineer(team: str, urgency: str) -> str:
    """Page the on-call engineer for a given team with a given urgency level."""
    return f"Paged {team} at {urgency}"


@tool
def create_incident(title: str, severity: str) -> str:
    """Create a formal incident record for tracking a major outage."""
    return f"Created incident: {title}"


@tool
def rollback_deployment(service_name: str, version: str) -> str:
    """Roll back a service deployment to a previous known-good version."""
    return f"Rolled back {service_name} to {version}"


@tool
def check_error_budget(service_name: str) -> str:
    """Check the remaining error budget for a service this quarter."""
    return f"Error budget for {service_name}"


@tool
def query_metrics_dashboard(metric_name: str) -> str:
    """Query a specific metric from the observability dashboard."""
    return f"Metric data for {metric_name}"


@tool
def send_customer_update(ticket_id: str, message: str) -> str:
    """Send a status update message to the customer on a ticket."""
    return f"Sent update on {ticket_id}"


@tool
def close_ticket(ticket_id: str, resolution: str) -> str:
    """Close a support ticket with a final resolution note."""
    return f"Closed {ticket_id}"


all_tools = [
    search_knowledge_base,
    get_ticket_status,
    check_service_health,
    compute_sla_breach,
    restart_service,
    page_oncall_engineer,
    create_incident,
    rollback_deployment,
    check_error_budget,
    query_metrics_dashboard,
    send_customer_update,
    close_ticket,
]


# -------------------------------------------------
# Index Tool Descriptions Into The Same Collection
# -------------------------------------------------

tool_documents = [
    Document(
        page_content=f"{tool_object.name}: {tool_object.description}",
        metadata={
            "type": "tool_description",
            "tool_name": tool_object.name,
        },
    )
    for tool_object in all_tools
]

vector_store.add_documents(tool_documents)

print("\n" + "=" * 60)
print(f"INDEXED {len(tool_documents)} TOOL DESCRIPTIONS")
print("=" * 60)


# -------------------------------------------------
# Retrieve Only Relevant Tool Descriptions
# -------------------------------------------------

def select_relevant_tools(
    question: str,
    top_k: int = 3,
) -> list:

    tool_retriever = vector_store.as_retriever(
        search_kwargs={
            "k": top_k,
            "filter": {
                "type": "tool_description",
            },
        }
    )

    matched_documents = tool_retriever.invoke(question)

    matched_names = {
        document.metadata["tool_name"]
        for document in matched_documents
    }

    return [
        tool_object
        for tool_object in all_tools
        if tool_object.name in matched_names
    ]


# -------------------------------------------------
# Test Retrieval Against A Sample Question
# -------------------------------------------------

question = "The billing service seems slow, what should I do?"

selected_tools = select_relevant_tools(question)

print(f"\nQuestion: {question}")
print(f"Selected tools ({len(selected_tools)} of {len(all_tools)}):")

for tool_object in selected_tools:
    print(f"  - {tool_object.name}")
