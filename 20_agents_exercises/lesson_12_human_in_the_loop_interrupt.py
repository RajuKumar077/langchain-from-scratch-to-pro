"""
Module 20 - Agent Architecture
Lesson 12 - Human-in-the-Loop Interrupt

Goal
----
- Pause the loop before any tool call actually executes
- Let a human approve or reject before the agent proceeds
"""

from typing import Annotated
from operator import add

from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.memory import MemorySaver


# -------------------------------------------------
# Tool: Restart Service (Potentially Destructive)
# -------------------------------------------------

@tool
def restart_service(
    service_name: str,
) -> str:
    """Restart a named backend service that is degraded or down."""

    return f"Restarted {service_name} successfully."


# -------------------------------------------------
# Create Chat Model And Bind Tools
# -------------------------------------------------

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

tools = [
    restart_service,
]

llm_with_tools = llm.bind_tools(tools)


# -------------------------------------------------
# State
# -------------------------------------------------

class State(TypedDict):

    messages: Annotated[list[AnyMessage], add]


SYSTEM_PROMPT = SystemMessage(content="""You are an on-call resolution
agent. Use the available tools to resolve infrastructure issues.""")


# -------------------------------------------------
# Node: Model
# -------------------------------------------------

def call_model(
    state: State,
) -> dict:

    messages = [SYSTEM_PROMPT] + state["messages"]

    response = llm_with_tools.invoke(messages)

    return {
        "messages": [response],
    }


# -------------------------------------------------
# Node: Tools
# -------------------------------------------------

tool_node = ToolNode(tools)


# -------------------------------------------------
# Build Graph
# -------------------------------------------------

graph_builder = StateGraph(State)

graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "model")

graph_builder.add_conditional_edges(
    "model",
    tools_condition,
)

graph_builder.add_edge("tools", "model")


# -------------------------------------------------
# Compile With A Checkpointer And An Interrupt Point
# -------------------------------------------------

checkpointer = MemorySaver()

agent = graph_builder.compile(
    checkpointer=checkpointer,
    interrupt_before=["tools"],
)


# -------------------------------------------------
# Run Until The Interrupt
# -------------------------------------------------

config = {
    "configurable": {
        "thread_id": "oncall-session-1",
    },
}

result = agent.invoke(
    {
        "messages": [
            HumanMessage(content="Restart billing-api, it's degraded."),
        ],
    },
    config=config,
)


# -------------------------------------------------
# Show The Pending Call And Ask For Approval
# -------------------------------------------------

pending_call = result["messages"][-1].tool_calls[0]

print("\n" + "=" * 60)
print("HUMAN-IN-THE-LOOP CHECKPOINT")
print("=" * 60)

print(f"\nAgent wants to call: {pending_call['name']}({pending_call['args']})")

approval = input("Approve? (y/n): ")


# -------------------------------------------------
# Resume Or Abort Based On Human Approval
# -------------------------------------------------

if approval.lower() == "y":

    result = agent.invoke(None, config=config)

    print("\n" + "=" * 60)
    print("RESUMED AFTER APPROVAL")
    print("=" * 60)

    print(result["messages"][-1].content)

else:

    print("\nTool call rejected - agent halted before execution.")
