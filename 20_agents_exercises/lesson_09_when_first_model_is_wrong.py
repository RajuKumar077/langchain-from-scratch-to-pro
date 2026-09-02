"""
Module 20 - Agent Architecture
Lesson 09 - When first_model Is Wrong

Goal
----
- Run the always-call-first agent against a question the KB cannot answer
- Observe the wasted step and confused reasoning it introduces
"""

import uuid
from typing import Annotated
from operator import add

from typing_extensions import TypedDict

from langchain_core.tools import tool
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, HumanMessage
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition


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


# -------------------------------------------------
# Tool: Search Knowledge Base
# -------------------------------------------------

@tool
def search_knowledge_base(
    query: str,
) -> str:
    """Search the internal knowledge base for troubleshooting
    articles and past ticket resolutions."""

    documents = retriever.invoke(query)

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# -------------------------------------------------
# Create Chat Model And Bind Tools
# -------------------------------------------------

llm = ChatOllama(
    model="qwen2.5:7b",
    temperature=0,
)

tools = [
    search_knowledge_base,
]

llm_with_tools = llm.bind_tools(tools)


# -------------------------------------------------
# State
# -------------------------------------------------

class State(TypedDict):

    messages: Annotated[list[AnyMessage], add]


SYSTEM_PROMPT = SystemMessage(content="""You are an on-call resolution
agent for BizOps support tickets.""")


# -------------------------------------------------
# Node: Always Search KB First (No LLM Call, No Judgment)
# -------------------------------------------------

def first_model(
    state: State,
) -> dict:

    last_human_message = state["messages"][-1]

    fabricated_call = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "search_knowledge_base",
                "args": {
                    "query": last_human_message.content,
                },
                "id": str(uuid.uuid4()),
            }
        ],
    )

    return {
        "messages": [fabricated_call],
    }


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

graph_builder.add_node("first_model", first_model)
graph_builder.add_node("model", call_model)
graph_builder.add_node("tools", tool_node)

graph_builder.add_edge(START, "first_model")
graph_builder.add_edge("first_model", "tools")
graph_builder.add_edge("tools", "model")
graph_builder.add_conditional_edges("model", tools_condition)

agent = graph_builder.compile()


# -------------------------------------------------
# A Question Where Forced KB Search Wastes A Step
# -------------------------------------------------

result = agent.invoke({
    "messages": [
        HumanMessage(content="What's 15% of a $240 invoice?"),
    ],
})


# -------------------------------------------------
# Display Full Conversation - Notice The Wasted First Step
# -------------------------------------------------

print("\n" + "=" * 60)
print("FIRST_MODEL MISAPPLIED TO A NON-KB QUESTION")
print("=" * 60)

for message in result["messages"]:

    print(f"\n[{message.type.upper()}]")
    if getattr(message, "tool_calls", None):
        print(f"  tool_calls: {message.tool_calls}")
    print(f"  content: {message.content}")
