"""
Module 21 - LangGraph Cognitive Architectures
Lesson 09 - LLM-Driven Routing Decision

Goal
----
- Fix brittle string parsing with a structured Pydantic output
- See rung 1 (LLM decides category) feed directly into rung 2 (category decides path)
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

    category: str  # one of: billing, access, technical, unknown


structured_llm = llm.with_structured_output(TicketCategory)


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
# Node: LLM Classifier (Rung 1, Now Structured)
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
# Node: Ask Clarifying Question
# -------------------------------------------------

def ask_clarifying_question(
    state: State,
) -> dict:

    return {
        "response": "Could you clarify whether this is a billing, access, or technical issue?",
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
# Router Function (Reads Structured Output Reliably)
# -------------------------------------------------

def route_by_category(
    state: State,
) -> str:

    category = state["category"]

    if category not in ("billing", "access", "technical"):
        return "clarify"

    if category == "billing":
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
graph_builder.add_node("ask_clarifying_question", ask_clarifying_question)
graph_builder.add_node("log_decision", log_decision)

graph_builder.add_edge(START, "classify_ticket")

graph_builder.add_conditional_edges(
    "classify_ticket",
    route_by_category,
    {
        "kb_lookup": "retrieve_context",
        "escalate": "escalate_to_human",
        "clarify": "ask_clarifying_question",
    },
)

graph_builder.add_edge("retrieve_context", "respond_to_ticket")
graph_builder.add_edge("respond_to_ticket", "log_decision")
graph_builder.add_edge("escalate_to_human", "log_decision")
graph_builder.add_edge("ask_clarifying_question", END)
graph_builder.add_edge("log_decision", END)

graph = graph_builder.compile()


# -------------------------------------------------
# Run Graph
# -------------------------------------------------

result = graph.invoke({
    "ticket_text": "I cannot log into my account after the password reset.",
})


# -------------------------------------------------
# Display Result
# -------------------------------------------------

print("\n" + "=" * 60)
print("STRUCTURED CLASSIFICATION + ROUTING (RUNG 1 -> RUNG 2)")
print("=" * 60)

print(f"Category: {result['category']}")
print(f"Response: {result['response']}")
