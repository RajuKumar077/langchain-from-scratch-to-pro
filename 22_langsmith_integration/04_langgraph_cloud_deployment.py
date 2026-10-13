"""
Lesson 04 - Production LangGraph Deployment Entrypoint (Local Ollama / Local Docker)
====================================================================================
Goal:
  1. Define a production-ready LangGraph graph using local Ollama (`qwen3:8b`).
  2. Export the compiled `graph` instance so that LangGraph CLI or LangSmith Deployment 
     can host it as a REST API endpoint locally or on server containers.

Deployment CLI Setup:
  pip install "langgraph-cli[inmem]"
  langgraph dev
"""

from typing import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage

# ------------------------------------------------------------------------------
# 1. State Definition
# ------------------------------------------------------------------------------
class DeploymentState(TypedDict):
    messages: Annotated[list[BaseMessage], operator.add]


# ------------------------------------------------------------------------------
# 2. Agent Node (Local Ollama qwen3:8b)
# ------------------------------------------------------------------------------
def assistant_node(state: DeploymentState) -> dict:
    """Core AI Assistant node processing messages via local Ollama."""
    llm = ChatOllama(model="qwen3:8b", temperature=0.5)
    
    # System prompt ensuring proper role definition
    sys_msg = SystemMessage(content="You are a helpful, professional assistant running locally via Ollama & LangGraph with full LangSmith tracing.")
    
    full_messages = [sys_msg] + state["messages"]
    response = llm.invoke(full_messages)
    
    return {"messages": [response]}


# ------------------------------------------------------------------------------
# 3. Construct Graph
# ------------------------------------------------------------------------------
builder = StateGraph(DeploymentState)
builder.add_node("assistant", assistant_node)
builder.add_edge(START, "assistant")
builder.add_edge("assistant", END)

# Export graph variable for langgraph.json
graph = builder.compile()

if __name__ == "__main__":
    print("[INIT] Production LangGraph Agent ready with local Ollama (qwen3:8b).")
    print("Test invoke locally:")
    res = graph.invoke({"messages": [HumanMessage(content="Hello! Is this local agent ready for deployment?")]})
    print("Response:", res["messages"][-1].content)
