"""
Module 20 - Agent Architecture
Lesson 08 - Always-Call-First Node

Goal
----
- Force the agent to always check the knowledge base first
- Fabricate a ToolCall by hand, without asking the LLM
- Prove the loop underneath is untouched
"""

import uuid
from typing import Annotated
from operator import add

from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage
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
    }

    return fake_tickets.get(ticket_id, "Ticket ID not found.")


# -------------------------------------------------
# Create Chat Model And Bind Tools
# -------------------------------------------------

llm = ChatOllama(
    model="qwen2.5:7b",
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


SYSTEM_PROMPT = SystemMessage(content="""You are an on-call resolution
agent for BizOps support tickets. Use the available tools to
investigate before answering.""")


# -------------------------------------------------
# Node: Always Search KB First (No LLM Call)
# -------------------------------------------------

def first_model(
    state: State,
) -> dict:

    last_human_message = state["messages"][-1]

    fabricated_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {
                    "query": last_human_message.content,
                },
                "id": str(uuid.uuid4()),
            }
        ],
    )

    return {
        "messages": [fabricated_call],
    }


# -------------------------------------------------
# Node: Model (The Unchanged Loop Node)
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
# Build Graph: START -> first_model -> tools -> model -> ... loop ...
# -------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("first_model", first_model)
graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "first_model")
graph_builder.add_edge("first_model", "tools")
graph_builder.add_edge("tools", "model")
graph_builder.add_conditional_edges("model", tools_condition)

agent = graph_builder.compile()


# -------------------------------------------------
# Run The Agent
# -------------------------------------------------

result = agent.invoke({
    "messages": [
        HumanMessage(content="How do I resolve a duplicate invoice charge?"),
    ],
})


# -------------------------------------------------
# Display Full Conversation
# -------------------------------------------------

print("\n" + "=" * 60)
print("ALWAYS-CALL-FIRST AGENT")
print("=" * 60)

for message in result["messages"]:

    print(f"\n[{message.type.upper()}]")
    if getattr(message, "tool_calls", None):
        print(f"  tool_calls: {message.tool_calls}")
    print(f"  content: {message.content}")
