"""
Module 20 - Agent Architecture
Lesson 06 - Full Multi-Tool ReAct Run

Goal
----
- Bind all four on-call tools
- Ask a question that requires chaining three tool calls before answering
- Watch the model decide its own stop condition
"""

from typing import Annotated
from operator import add

from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


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
    articles and past ticket resolutions."""

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
# Create Chat Model And Bind All Four Tools
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)

tools = [
    search_knowledge_base,
    get_ticket_status,
    check_service_health,
    compute_sla_breach,
]

llm_with_tools = llm.bind_tools(tools)


# -------------------------------------------------
# State
# -------------------------------------------------

class State(TypedDict):

    messages: Annotated[list[AnyMessage], add]


# -------------------------------------------------
# System Prompt
# -------------------------------------------------

SYSTEM_PROMPT = SystemMessage(content="""You are an on-call resolution
agent for BizOps support tickets. Use the available tools to
investigate before answering. Only stop once you have enough
information to give a concrete resolution or escalation decision.""")


# -------------------------------------------------
# Node: Model
# -------------------------------------------------

def call_model(
    state: State,
) -> dict:

    messages = [SYSTEM_PROMPT] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
    }


# -------------------------------------------------
# Node: Tools
# -------------------------------------------------

tool_node = ToolNode(tools)


# -------------------------------------------------
# Build Graph
# -------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "model")

graph_builder.add_conditional_edges(
    "model",
    tools_condition,
)

graph_builder.add_edge("tools", "model")

agent = graph_builder.compile()


# -------------------------------------------------
# Run The Agent On A Multi-Step Question
# -------------------------------------------------

result = agent.invoke({
    "messages": [
        HumanMessage(content="Ticket T-1001 has been open a while, is billing-api healthy, and has it breached SLA?"),
    ],
})


# -------------------------------------------------
# Display Full Conversation
# -------------------------------------------------

print("\n" + "=" * 60)
print("MULTI-STEP AGENT CONVERSATION")
print("=" * 60)

for message in result["messages"]:

    print(f"\n[{message.type.upper()}]")
    if getattr(message, "tool_calls", None):
        print(f"  tool_calls: {message.tool_calls}")
    print(f"  content: {message.content}")
