"""
Module 21 - LangGraph Cognitive Architectures
Lesson 01 - Rung 0: A Graph With No LLM

Goal
----
- Define a TypedDict State
- Write a plain-Python node (no LLM call)
- Wire START -> node -> END
- Compile and invoke the graph
"""

from typing import TypedDict

from langgraph.graph import StateGraph, START, END


# -------------------------------------------------
# Define State
# -------------------------------------------------

class State(TypedDict):

    ticket_text: str
    category: str


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
# Build Graph
# -------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node(
    "classify_ticket",
    classify_ticket,
)

graph_builder.add_edge(START, "classify_ticket")
graph_builder.add_edge("classify_ticket", END)

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
print("TICKET CLASSIFICATION (RUNG 0)")
print("=" * 60)

print(f"Ticket:   {ticket_text}")
print(f"Category: {result['category']}")


# -------------------------------------------------
# Visualizations
# -------------------------------------------------

print("\n" + "=" * 60)
print("GRAPH VISUALIZATION (MERMAID TEXT)")
print("=" * 60)

# Option A: Print Mermaid syntax (paste this into GitHub/VS Code Markdown)
print(graph.get_graph().draw_mermaid())


# Option B: Save directly to a PNG image file
try:
    png_bytes = graph.get_graph().draw_mermaid_png()
    with open("ticket_graph.png", "wb") as f:
        f.write(png_bytes)
    print("\n[SUCCESS] Graph visualization saved as 'ticket_graph.png'")
except Exception as e:
    print(f"\n[NOTE] Could not generate PNG file automatically: {e}")
    print("Ensure internet connection or pillow library is installed.")

