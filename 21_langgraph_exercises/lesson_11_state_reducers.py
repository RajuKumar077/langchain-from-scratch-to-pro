"""
Module 21 - LangGraph Cognitive Architectures
Lesson 11 - State Reducers: Accumulating An Audit Trail

Goal
----
- Replace print-statement logging with a real audit_trail field
- Use Annotated[list, add] so every node's write is combined, not overwritten
"""

from typing import TypedDict, Annotated
from operator import add

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
# State With A Reducer Field
# -------------------------------------------------

class State(TypedDict):

    ticket_text: str
    category: str
    retrieved_context: str
    response: str
    audit_trail: Annotated[list[str], add]


# -------------------------------------------------
# Node: LLM Classifier (Appends To Audit Trail)
# -------------------------------------------------

def classify_ticket(
    state: State,
) -> dict:

    result = structured_llm.invoke(
        f"Classify this ticket into billing, access, technical, or unknown: {state['ticket_text']}"
    )

    return {
        "category": result.category,
        "audit_trail": [f"classify_ticket: category={result.category}"],
    }


# -------------------------------------------------
# Node: Retrieve Context (Appends To Audit Trail)
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
        "audit_trail": [f"retrieve_context: {len(documents)} documents retrieved"],
    }


# -------------------------------------------------
# Node: Respond To Ticket (Appends To Audit Trail)
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
        "audit_trail": ["respond_to_ticket: response drafted"],
    }


# -------------------------------------------------
# Node: Escalate To Human (Appends To Audit Trail)
# -------------------------------------------------

def escalate_to_human(
    state: State,
) -> dict:

    return {
        "response": "This ticket has been escalated to a human agent.",
        "audit_trail": ["escalate_to_human: escalated"],
    }


# -------------------------------------------------
# Node: Audit Logger (Final Entry)
# -------------------------------------------------

def log_decision(
    state: State,
) -> dict:

    return {
        "audit_trail": ["log_decision: final response logged"],
    }


# -------------------------------------------------
# Router Function
# -------------------------------------------------

def route_by_category(
    state: State,
) -> str:

    category = state["category"]

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
graph_builder.add_node("log_decision", log_decision)

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
graph_builder.add_edge("respond_to_ticket", "log_decision")
graph_builder.add_edge("escalate_to_human", "log_decision")
graph_builder.add_edge("log_decision", END)

graph = graph_builder.compile()


# -------------------------------------------------
# Run Graph
# -------------------------------------------------

result = graph.invoke({
    "ticket_text": "The app keeps crashing on startup.",
    "audit_trail": [],
})


# -------------------------------------------------
# Display The Full Accumulated Audit Trail
# -------------------------------------------------

print("\n" + "=" * 60)
print("ACCUMULATED AUDIT TRAIL (REDUCER IN ACTION)")
print("=" * 60)

for entry in result["audit_trail"]:
    print(f"- {entry}")
