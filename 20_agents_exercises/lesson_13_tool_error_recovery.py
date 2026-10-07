"""
Module 20 - Agent Architecture
Lesson 13 - Tool Error Recovery

Goal
----
- Make check_service_health raise for an unknown service
- See ToolNode surface the failure back into the loop instead of crashing it
"""

from typing import Annotated
from operator import add

from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


# -------------------------------------------------
# Tool That Can Genuinely Fail
# -------------------------------------------------

@tool
def check_service_health(
    service_name: str,
) -> str:
    """Check whether a named backend service is currently healthy,
    degraded, or down. Raises if the service is not registered."""

    fake_health = {
        "billing-api": "DEGRADED - elevated latency since 09:00 UTC",
        "auth-service": "HEALTHY",
    }

    if service_name not in fake_health:
        raise ValueError(f"Unknown service: {service_name}")

    return fake_health[service_name]


# -------------------------------------------------
# Create Chat Model And Bind Tools
# -------------------------------------------------

llm = ChatOllama(
    model="qwen3:8b",
    temperature=0,
)

tools = [
    check_service_health,
]

llm_with_tools = llm.bind_tools(tools)


# -------------------------------------------------
# State
# -------------------------------------------------

class State(TypedDict):

    messages: Annotated[list[AnyMessage], add]


SYSTEM_PROMPT = SystemMessage(content="""You are an on-call resolution
agent. If a tool call fails, try a corrected input or explain the failure
to the user.""")


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
# Node: Tools With Error Handling Enabled (Default Behavior)
# -------------------------------------------------

tool_node = ToolNode(
    tools,
    handle_tool_errors=True,
)


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

agent = graph_builder.compile()


# -------------------------------------------------
# Run The Agent Against An Unregistered Service Name
# -------------------------------------------------

result = agent.invoke({
    "messages": [
        HumanMessage(content="Is foo-bar-service healthy?"),
    ],
})


# -------------------------------------------------
# Display Full Conversation - Note The Error Recovers Gracefully
# -------------------------------------------------

print("\n" + "=" * 60)
print("TOOL ERROR RECOVERY TEST")
print("=" * 60)

for message in result["messages"]:

    print(f"\n[{message.type.upper()}]")
    if getattr(message, "tool_calls", None):
        print(f"  tool_calls: {message.tool_calls}")
    print(f"  content: {message.content}")
