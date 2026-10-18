"""
Module 13 - LangGraph Cognitive Architectures
Lesson 05 - Input / Output Schemas

Goal
----
- Separate what a caller passes in from what they get back
- Hide internal fields from the public output
"""

from typing import TypedDict

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
# Split State Into Input / Internal / Output
# -------------------------------------------------

class InputState(TypedDict):

    ticket_text: str


class OutputState(TypedDict):

    category: str
    response: str


class State(InputState, OutputState):

    retrieved_context: str
    logged: bool


# -------------------------------------------------
# Node: LLM Classifier
# -------------------------------------------------

def classify_ticket(
    state: State,
) -> dict:

    prompt = f"""Classify this support ticket into exactly one word:
billing, access, or technical.

Ticket: {state['ticket_text']}

Respond with only the single word."""

    response = llm.invoke(prompt)

    return {
        "category": response.content.strip().lower(),
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
# Node: Audit Logger
# -------------------------------------------------

def log_decision(
    state: State,
) -> dict:

    print(f"[AUDIT] category={state['category']}")

    return {
        "logged": True,
    }


# -------------------------------------------------
# Build Graph With Declared Input / Output
# -------------------------------------------------

graph_builder = StateGraph(
    State,
    input=InputState,
    output=OutputState,
)

graph_builder.add_node("classify_ticket", classify_ticket)
graph_builder.add_node("retrieve_context", retrieve_context)
graph_builder.add_node("respond_to_ticket", respond_to_ticket)
graph_builder.add_node("log_decision", log_decision)

graph_builder.add_edge(START, "classify_ticket")
graph_builder.add_edge("classify_ticket", "retrieve_context")
graph_builder.add_edge("retrieve_context", "respond_to_ticket")
graph_builder.add_edge("respond_to_ticket", "log_decision")
graph_builder.add_edge("log_decision", END)

graph = graph_builder.compile()


# -------------------------------------------------
# Caller Only Sees InputState In, OutputState Out
# -------------------------------------------------

result = graph.invoke({
    "ticket_text": "My invoice shows a duplicate charge.",
})


# -------------------------------------------------
# Display Result
# -------------------------------------------------

print("\n" + "=" * 60)
print("FILTERED OUTPUT (INPUT/OUTPUT SCHEMA)")
print("=" * 60)

print(result)
print("\nKeys returned:", list(result.keys()))
