"""
Module 21 - LangGraph Cognitive Architectures
Lesson 12 - Streaming + Auto-Diagram

Goal
----
- Use graph.stream() to observe execution node-by-node
- Generate the Mermaid diagram straight from the compiled graph object
"""

from typing import TypedDict

from pydantic import BaseModel
from langgraph.graph import StateGraph, START, END
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

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# -------------------------------------------------
# Structured Category Schema
# -------------------------------------------------

class TicketCategory(BaseModel):

    category: str


structured_llm = llm.with_structured_output(TicketCategory)


# -------------------------------------------------
# Define State
# -------------------------------------------------

class State(TypedDict):

    ticket_text: str
    category: str
    retrieved_context: str
    response: str


# -------------------------------------------------
# Node: LLM Classifier
# -------------------------------------------------

def classify_ticket(
    state: State,
) -> dict:

    result = structured_llm.invoke(
        f"Classify this ticket into billing, access, technical, or unknown: {state['ticket_text']}"
    )

    return {
        "category": result.category,
    }


# -------------------------------------------------
# Node: Retrieve Context
# -------------------------------------------------

def retrieve_context(
    state: State,
) -> dict:

    documents = retriever.invoke(state["ticket_text"])

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return {
        "retrieved_context": context,
    }


# -------------------------------------------------
# Node: Respond To Ticket
# -------------------------------------------------

def respond_to_ticket(
    state: State,
) -> dict:

    prompt = f"""Category: {state['category']}
Context:
{state['retrieved_context']}

Ticket: {state['ticket_text']}

Write a short resolution reply using only the context above."""

    response = llm.invoke(prompt)

    return {
        "response": response.content,
    }


# -------------------------------------------------
# Node: Escalate To Human
# -------------------------------------------------

def escalate_to_human(
    state: State,
) -> dict:

    return {
        "response": "This ticket has been escalated to a human agent.",
    }


# -------------------------------------------------
# Router Function
# -------------------------------------------------

def route_by_category(
    state: State,
) -> str:

    if state["category"] == "billing":
        return "escalate"

    return "kb_lookup"


# -------------------------------------------------
# Build Graph
# -------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("classify_ticket", classify_ticket)
graph_builder.add_node("retrieve_context", retrieve_context)
graph_builder.add_node("respond_to_ticket", respond_to_ticket)
graph_builder.add_node("escalate_to_human", escalate_to_human)

graph_builder.add_edge(START, "classify_ticket")

graph_builder.add_conditional_edges(
    "classify_ticket",
    route_by_category,
    {
        "kb_lookup": "retrieve_context",
        "escalate": "escalate_to_human",
    },
)

graph_builder.add_edge("retrieve_context", "respond_to_ticket")
graph_builder.add_edge("respond_to_ticket", END)
graph_builder.add_edge("escalate_to_human", END)

graph = graph_builder.compile()


# -------------------------------------------------
# Auto-Generate The Diagram From The Compiled Graph
# -------------------------------------------------

print("\n" + "=" * 60)
print("AUTO-GENERATED MERMAID DIAGRAM")
print("=" * 60)

print(graph.get_graph().draw_mermaid())


# -------------------------------------------------
# Stream Execution Node-By-Node
# -------------------------------------------------

print("\n" + "=" * 60)
print("STREAMED EXECUTION")
print("=" * 60)

for step in graph.stream({
    "ticket_text": "My invoice shows a duplicate charge.",
}):

    for node_name, node_output in step.items():

        print(f"\n[{node_name}] -> {node_output}")
