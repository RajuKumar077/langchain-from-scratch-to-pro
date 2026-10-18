"""
Module 12 - Agent Architecture
Lesson 14 - Autonomy Step-Down, Proven

Goal
----
- Build the plain ReAct loop and the fully extended loop in one file
- Programmatically diff their model/tools edges
- Prove both extensions only prepended nodes and never touched the loop
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
# Shared Tool
# -------------------------------------------------

@tool
def search_knowledge_base(
    query: str,
) -> str:
    """Search the internal knowledge base for troubleshooting articles."""

    documents = retriever.invoke(query)

    return "\n\n".join(
        document.page_content
        for document in documents
    )


tools = [search_knowledge_base]

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
)

llm_with_tools = llm.bind_tools(tools)


# -------------------------------------------------
# State
# -------------------------------------------------

class State(TypedDict):

    messages: Annotated[list[AnyMessage], add]


SYSTEM_PROMPT = SystemMessage(content="""You are an on-call resolution
agent for BizOps support tickets.""")


# -------------------------------------------------
# Shared Loop Nodes
# -------------------------------------------------

def call_model(
    state: State,
) -> dict:

    messages = [SYSTEM_PROMPT] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
    }


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


tool_node = ToolNode(tools)


# -------------------------------------------------
# Build The Plain Loop (Exercise 5's Shape)
# -------------------------------------------------

plain_builder = StateGraph(State)

plain_builder.add_node("model", call_model)
plain_builder.add_node("tools", tool_node)

plain_builder.add_edge(START, "model")
plain_builder.add_conditional_edges("model", tools_condition)
plain_builder.add_edge("tools", "model")

plain_agent = plain_builder.compile()


# -------------------------------------------------
# Build The Extended Loop (Extension 1 Prepended)
# -------------------------------------------------

extended_builder = StateGraph(State)

extended_builder.add_node("first_model", first_model)
extended_builder.add_node("model", call_model)
extended_builder.add_node("tools", tool_node)

extended_builder.add_edge(START, "first_model")
extended_builder.add_edge("first_model", "tools")
extended_builder.add_edge("tools", "model")
extended_builder.add_conditional_edges("model", tools_condition)

extended_agent = extended_builder.compile()


# -------------------------------------------------
# Print Both Diagrams
# -------------------------------------------------

print("\n" + "=" * 60)
print("PLAIN LOOP")
print("=" * 60)
print(plain_agent.get_graph().draw_mermaid())

print("\n" + "=" * 60)
print("EXTENDED LOOP")
print("=" * 60)
print(extended_agent.get_graph().draw_mermaid())


# -------------------------------------------------
# Assert The Loop Edges Are Unchanged
# -------------------------------------------------

plain_edges = {
    (edge.source, edge.target)
    for edge in plain_agent.get_graph().edges
    if edge.source in ("model", "tools")
}

extended_edges = {
    (edge.source, edge.target)
    for edge in extended_agent.get_graph().edges
    if edge.source in ("model", "tools")
}

print("\n" + "=" * 60)
print("LOOP EDGE COMPARISON")
print("=" * 60)

print(f"Plain loop edges:    {plain_edges}")
print(f"Extended loop edges: {extended_edges}")

assert plain_edges == extended_edges, "Loop structure changed - extension broke isolation!"

print("\nLoop edges identical across both graphs - extension only prepended a node.")
