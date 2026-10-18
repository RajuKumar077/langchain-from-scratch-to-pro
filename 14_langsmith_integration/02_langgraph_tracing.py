"""
Lesson 02 - Tracing LangGraph Workflows & State Machine Nodes
============================================================
Goal:
  1. Build a multi-step router agent using LangGraph.
  2. Ensure LangSmith environment setup runs BEFORE LangChain initialization.
  3. See state transitions and node executions in the LangSmith trace visualizer.

Run command:
  r:\\LangChain-Book\\.venv\\Scripts\\python.exe 02_langgraph_tracing.py
"""

# CRITICAL: Import config FIRST to set os.environ before LangChain initializes!
import config

import sys
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langsmith import traceable, Client

# ------------------------------------------------------------------------------
# 1. Define State Schema
# ------------------------------------------------------------------------------
class AgentState(TypedDict):
    query: str
    category: str 
    reasoning: str
    response: str
    logs: List[str]


# ------------------------------------------------------------------------------
# 2. Graph Nodes
# ------------------------------------------------------------------------------
@traceable(name="Analyze & Route Node", run_type="chain")
def route_query_node(state: AgentState) -> dict:
    query = state["query"].lower()
    logs = state.get("logs", [])
    logs.append("Query analyzed by classifier node.")

    if any(k in query for k in ["bug", "error", "exception", "failed"]):
        category = "technical_support"
        reasoning = "Detected error keywords in query."
    elif any(k in query for k in ["price", "billing", "invoice", "refund"]):
        category = "billing"
        reasoning = "Detected financial/billing keywords."
    else:
        category = "general"
        reasoning = "No specialized keywords matched, defaulting to general."

    return {"category": category, "reasoning": reasoning, "logs": logs}


@traceable(name="Tech Support Handler Node", run_type="tool")
def tech_support_node(state: AgentState) -> dict:
    return {
        "response": f"[Tech Support Desk]: We have logged your technical issue ('{state['query']}'). Check system logs for stack traces.",
        "logs": state.get("logs", []) + ["Routed to Tech Support."]
    }


@traceable(name="Billing Handler Node", run_type="tool")
def billing_node(state: AgentState) -> dict:
    return {
        "response": f"[Billing Desk]: Please visit account billing settings to review charges for: '{state['query']}'.",
        "logs": state.get("logs", []) + ["Routed to Billing."]
    }


@traceable(name="General Handler Node", run_type="tool")
def general_node(state: AgentState) -> dict:
    return {
        "response": f"[General Desk]: Thank you for reaching out! Regarding: '{state['query']}', how can we assist further?",
        "logs": state.get("logs", []) + ["Routed to General Assistant."]
    }


def router_conditional_edge(state: AgentState) -> str:
    """Conditional router function determining target node based on category."""
    return state["category"]


# ------------------------------------------------------------------------------
# 3. Build & Compile LangGraph Graph
# ------------------------------------------------------------------------------
def build_agent_graph():
    builder = StateGraph(AgentState)

    # Add Nodes
    builder.add_node("classifier", route_query_node)
    builder.add_node("technical_support", tech_support_node)
    builder.add_node("billing", billing_node)
    builder.add_node("general", general_node)

    # Connect Edges
    builder.add_edge(START, "classifier")
    
    # Conditional edge routing
    builder.add_conditional_edges(
        "classifier",
        router_conditional_edge,
        {
            "technical_support": "technical_support",
            "billing": "billing",
            "general": "general"
        }
    )

    builder.add_edge("technical_support", END)
    builder.add_edge("billing", END)
    builder.add_edge("general", END)

    return builder.compile()


# ------------------------------------------------------------------------------
# 4. Main Execution
# ------------------------------------------------------------------------------
def main():
    graph = build_agent_graph()

    test_queries = [
        "I got a NullPointer exception when clicking submit",
        "Can I get a copy of my monthly invoice?",
        "What are your business hours?" 
    ]

    print("[START] Invoking LangGraph agent workflow with LangSmith tracing enabled...\n")

    for idx, q in enumerate(test_queries, 1):
        print(f"--- Query #{idx}: '{q}' ---")
        final_state = graph.invoke(
            {"query": q, "logs": []},
            config={
                "run_name": f"LangGraph Support Ticket Routing #{idx}",
                "tags": ["langgraph", "support-bot", "routing-demo"],
                "metadata": {"test_case_id": f"TC-00{idx}"}
            }
        )
        print(f"Category : {final_state['category']}")
        print(f"Reasoning: {final_state['reasoning']}")
        print(f"Response : {final_state['response']}")
        print(f"Logs     : {final_state['logs']}\n")

    print("[FLUSHING TRACES] Synchronizing trace logs with LangSmith...")
    try:
        client = Client()
        client.flush()
    except Exception as e:
        print(f"[NOTE] Trace flush warning: {e}")

    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    print(f"[SUCCESS] All queries processed. View traces in project '{project_name}' on LangSmith Dashboard.")

if __name__ == "__main__":
    main()
