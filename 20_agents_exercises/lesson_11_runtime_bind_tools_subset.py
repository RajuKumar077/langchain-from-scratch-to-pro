"""
Module 20 - Agent Architecture
Lesson 11 - Runtime bind_tools With a Subset

Goal
----
- Prepend a select_tools node that retrieves a relevant tool subset
- Bind the LLM to only that subset, per-question, inside the graph
- Prove the model -> tools -> model loop is unchanged
"""

from typing import Annotated
from operator import add

from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.documents import Document
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
# Real Tools
# -------------------------------------------------

@tool
def search_knowledge_base(query: str) -> str:
    """Search the internal knowledge base for troubleshooting articles."""

    documents = retriever.invoke(query)

    return "\n\n".join(
        document.page_content
        for document in documents
    )


@tool
def check_service_health(service_name: str) -> str:
    """Check whether a named backend service is healthy, degraded, or down."""

    fake_health = {
        "billing-api": "DEGRADED - elevated latency since 09:00 UTC",
    }

    return fake_health.get(service_name, "Unknown service.")


@tool
def restart_service(service_name: str) -> str:
    """Restart a named backend service that is degraded or down."""

    return f"Restarted {service_name} successfully."


@tool
def compute_sla_breach(hours_open: float, priority: str) -> str:
    """Determine whether a ticket has breached its SLA window."""

    sla_hours = {"P1": 4, "P2": 24, "P3": 72}
    limit = sla_hours.get(priority.upper(), 72)

    if hours_open > limit:
        return f"SLA BREACHED - open {hours_open}h, limit {limit}h"

    return f"Within SLA - open {hours_open}h, limit {limit}h"


all_tools = [
    search_knowledge_base,
    check_service_health,
    restart_service,
    compute_sla_breach,
]


# -------------------------------------------------
# Index Tool Descriptions (Idempotent - Safe To Rerun)
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


# -------------------------------------------------
# Retrieve Only Relevant Tool Descriptions
# -------------------------------------------------

def select_relevant_tools(
    question: str,
    top_k: int = 2,
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
# Create Chat Model (Unbound - Binding Happens Per-Question)
# -------------------------------------------------

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
)


# -------------------------------------------------
# State: Carries The Selected Tool Subset
# -------------------------------------------------

class State(TypedDict):

    messages: Annotated[list[AnyMessage], add]
    available_tools: list


SYSTEM_PROMPT = SystemMessage(content="""You are an on-call resolution
agent for BizOps support tickets. Use only the tools available to you.""")


# -------------------------------------------------
# Node: Select Tools (Prepended, Rung 1 - No Path Change)
# -------------------------------------------------

def select_tools_node(
    state: State,
) -> dict:

    last_human_message = state["messages"][-1]

    selected = select_relevant_tools(
        last_human_message.content,
    )

    print(f"[select_tools] chose: {[t.name for t in selected]}")

    return {
        "available_tools": selected,
    }


# -------------------------------------------------
# Node: Model, Now Binding At Runtime
# -------------------------------------------------

def call_model(
    state: State,
) -> dict:

    scoped_llm = llm.bind_tools(state["available_tools"])

    messages = [SYSTEM_PROMPT] + state["messages"]

    response = scoped_llm.invoke(messages)

    return {
        "messages": [response],
    }


# -------------------------------------------------
# Node: Tools (Bound To The Full Catalog For Dispatch)
# -------------------------------------------------

tool_node = ToolNode(all_tools)


# -------------------------------------------------
# Build Graph: START -> select_tools -> model -> ... loop unchanged ...
# -------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("select_tools", select_tools_node)
graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "select_tools")
graph_builder.add_edge("select_tools", "model")
graph_builder.add_conditional_edges("model", tools_condition)
graph_builder.add_edge("tools", "model")

agent = graph_builder.compile()


# -------------------------------------------------
# Run The Agent
# -------------------------------------------------

result = agent.invoke({
    "messages": [
        HumanMessage(content="billing-api seems degraded, can you restart it?"),
    ],
    "available_tools": [],
})


# -------------------------------------------------
# Display Full Conversation
# -------------------------------------------------

print("\n" + "=" * 60)
print("SCOPED-TOOL AGENT RUN")
print("=" * 60)

for message in result["messages"]:

    print(f"\n[{message.type.upper()}]")
    if getattr(message, "tool_calls", None):
        print(f"  tool_calls: {message.tool_calls}")
    print(f"  content: {message.content}")
