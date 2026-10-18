"""
Module 13 - LangGraph Cognitive Architectures
Lesson 02 - Chaining Two Nodes (Still Rung 0)

Goal
----
- Add a second node, log_decision
- Chain classify_ticket -> log_decision -> END
- See that more nodes do NOT mean more autonomy
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# -------------------------------------------------
# Define State
# -------------------------------------------------

class State(TypedDict):

    ticket_text: str
    category: str
    logged: bool


# -------------------------------------------------
# Node: Keyword Classifier (Rung 0 - No LLM)
# -------------------------------------------------

def classify_ticket(
    state: State,
) -> dict:

    ticket_text = state["ticket_text"].lower()

    if "password" in ticket_text or "login" in ticket_text:
        category = "access"
    elif "invoice" in ticket_text or "charge" in ticket_text:
        category = "billing"
    else:
        category = "technical"

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
# Build Graph With Two Nodes
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

ticket_text = "I cannot log in, it says my password is wrong."

result = graph.invoke({
    "ticket_text": ticket_text,
})


# -------------------------------------------------
# Display Result
# -------------------------------------------------

print("\n" + "=" * 60)
print("TICKET CLASSIFICATION + AUDIT LOG (RUNG 0)")
print("=" * 60)

print(f"Ticket:   {ticket_text}")
print(f"Category: {result['category']}")
print(f"Logged:   {result['logged']}")
