"""
Module 13 - LangGraph Cognitive Architectures
Lesson 06 - A Third Chain Link: Summarization

Goal
----
- Insert a summarize_context node between retrieve and respond
- Understand chain length as a token-budget decision
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
# Define State
# -------------------------------------------------

class State(TypedDict):

    ticket_text: str
    category: str
    retrieved_context: str
    response: str
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
# Node: Summarize Context Before Responding
# -------------------------------------------------

def summarize_context(
    state: State,
) -> dict:

    prompt = f"""Compress the following context into 3 short bullet points,
keeping only information relevant to a support ticket:

{state['retrieved_context']}"""

    response = llm.invoke(prompt)

    return {
        "retrieved_context": response.content,
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
# Build Graph: classify -> retrieve -> summarize -> respond -> log -> END
# -------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("classify_ticket", classify_ticket)
graph_builder.add_node("retrieve_context", retrieve_context)
graph_builder.add_node("summarize_context", summarize_context)
graph_builder.add_node("respond_to_ticket", respond_to_ticket)
graph_builder.add_node("log_decision", log_decision)

graph_builder.add_edge(START, "classify_ticket")
graph_builder.add_edge("classify_ticket", "retrieve_context")
graph_builder.add_edge("retrieve_context", "summarize_context")
graph_builder.add_edge("summarize_context", "respond_to_ticket")
graph_builder.add_edge("respond_to_ticket", "log_decision")
graph_builder.add_edge("log_decision", END)

graph = graph_builder.compile()


# -------------------------------------------------
# Run Graph
# -------------------------------------------------

result = graph.invoke({
    "ticket_text": "My invoice shows a duplicate charge, can you help?",
})


# -------------------------------------------------
# Display Result
# -------------------------------------------------

print("\n" + "=" * 60)
print("FIVE-NODE CHAIN WITH SUMMARIZATION (RUNG 1)")
print("=" * 60)

print(f"Category: {result['category']}")
print(f"\nSummarized Context:\n{result['retrieved_context']}")
print(f"\nResponse:\n{result['response']}")
