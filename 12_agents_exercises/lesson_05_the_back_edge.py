"""
Module 12 - Agent Architecture
Lesson 05 - The Back-Edge: Closing the Loop
┌──────────────────────┐
                  ▼                      │
START ──> [model] ──(tools_condition)──> [tools]
            │
            └──(No tool calls)──> END
Goal
----
- Wire tools -> model - the edge that turns a chain into a loop
- Compile and run the full ReAct agent
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
    """Look up the current status of a support ticket by its ID."""

    fake_tickets = {
        "T-1001": "OPEN, priority=P2, last_updated=2h ago",
        "T-1002": "RESOLVED, priority=P3, last_updated=1d ago",
    }

    return fake_tickets.get(ticket_id, "Ticket ID not found.")


# -------------------------------------------------
# Create Chat Model And Bind Tools
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)

tools = [
    search_knowledge_base,
    get_ticket_status,
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
# Build Graph WITH The Back-Edge
# -------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "model")

graph_builder.add_conditional_edges(
    "model",
    tools_condition,
)

graph_builder.add_edge("tools", "model")  # THE BACK-EDGE

agent = graph_builder.compile()


# -------------------------------------------------
# Run The Agent
# -------------------------------------------------

result = agent.invoke({
    "messages": [
        HumanMessage(content="What's the status of ticket T-1001?"),
    ],
})


# -------------------------------------------------
# Display Full Conversation
# -------------------------------------------------

print("\n" + "=" * 60)
print("AGENT CONVERSATION (LOOP CLOSED)")
print("=" * 60)

for message in result["messages"]:

    print(f"\n[{message.type.upper()}]")
    if getattr(message, "tool_calls", None):
        print(f"  tool_calls: {message.tool_calls}")
    print(f"  content: {message.content}")
