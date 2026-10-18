"""
Module 13 - LangGraph Cognitive Architectures
Lesson 03 - Rung 1: LLM Decides a Step's Output

Goal
----
- Replace the keyword classifier with an LLM call
- Path stays fixed - only the output value is now LLM-produced
- See the exact rung 0 -> rung 1 boundary
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

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
    logged: bool


# -------------------------------------------------
# Node: LLM Classifier (Rung 1 - LLM Decides Output)
# -------------------------------------------------

def classify_ticket(
    state: State,
) -> dict:

    ticket_text = state["ticket_text"]

    prompt = f"""Classify this support ticket into exactly one word:
billing, access, or technical.

Ticket: {ticket_text}

Respond with only the single word."""

    response = llm.invoke(prompt)
    category = response.content.strip().lower()

    return {
        "category": category,
    }


# -------------------------------------------------
# Node: Audit Logger (Rung 0 - No LLM)
# -------------------------------------------------

def log_decision(
    state: State,
) -> dict:

    print(f"[AUDIT] category={state['category']}")

    return {
        "logged": True,
    }


# -------------------------------------------------
# Build Graph
# -------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("classify_ticket", classify_ticket)
graph_builder.add_node("log_decision", log_decision)

graph_builder.add_edge(START, "classify_ticket")
graph_builder.add_edge("classify_ticket", "log_decision")
graph_builder.add_edge("log_decision", END)

graph = graph_builder.compile()


# -------------------------------------------------
# Run Graph
# -------------------------------------------------

ticket_text = "I was charged twice for the same invoice this month."

result = graph.invoke({
    "ticket_text": ticket_text,
})


# -------------------------------------------------
# Display Result
# -------------------------------------------------

print("\n" + "=" * 60)
print("TICKET CLASSIFICATION (RUNG 1 - LLM OUTPUT)")
print("=" * 60)

print(f"Ticket:   {ticket_text}")
print(f"Category: {result['category']}")
print(f"Logged:   {result['logged']}")
