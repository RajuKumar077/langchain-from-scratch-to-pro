"""
Module 20 - Agent Architecture
Lesson 04 - ToolNode + tools_condition

Goal
----
- Build the model node that calls the bound LLM
- Add a prebuilt ToolNode that executes whatever tool call the model emits
- Route conditionally with tools_condition: tool call -> tools, else -> END
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
# State: A Running List Of Messages
# -------------------------------------------------

class State(TypedDict):

    messages: Annotated[list[AnyMessage], add]


# -------------------------------------------------
# System Prompt For The On-Call Agent
# -------------------------------------------------

SYSTEM_PROMPT = SystemMessage(content="""You are an on-call resolution
agent for BizOps support tickets. Use the available tools to
investigate before answering.""")


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
# Node: Tools (Prebuilt)
# -------------------------------------------------

tool_node = ToolNode(tools)


# -------------------------------------------------
# Build Graph (No Back-Edge Yet - One Tool Call Max)
# -------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "model")

graph_builder.add_conditional_edges(
    "model",
    tools_condition,
)

graph_builder.add_edge("tools", END)  # dead-ends after one tool call

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
# Display Conversation
# -------------------------------------------------

print("\n" + "=" * 60)
print("SINGLE TOOL CALL (NO LOOP YET)")
print("=" * 60)

for message in result["messages"]:

    print(f"\n[{message.type.upper()}]")
    if getattr(message, "tool_calls", None):
        print(f"  tool_calls: {message.tool_calls}")
    print(f"  content: {message.content}")
