"""
Module 20 - Agent Architecture
Lesson 02 - bind_tools and Reading a ToolCall

Goal
----
- Bind one tool to the LLM
- Inspect the raw ToolCall structure the model produces
"""

from langchain_core.tools import tool
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


# -------------------------------------------------
# Tool: Search Knowledge Base
# -------------------------------------------------

@tool
def search_knowledge_base(
    query: str,
) -> str:
    """Search the internal knowledge base for troubleshooting
    articles and past ticket resolutions. Use this when the
    user's question could be answered from documented procedures."""

    documents = retriever.invoke(query)

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# -------------------------------------------------
# Create Chat Model (qwen2.5:7b - reliable tool calling)
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# -------------------------------------------------
# Bind The Tool
# -------------------------------------------------

llm_with_tools = llm.bind_tools([
    search_knowledge_base,
])


# -------------------------------------------------
# Ask A Question That Should Trigger The Tool
# -------------------------------------------------

response = llm_with_tools.invoke(
    "How do I resolve a duplicate invoice charge ticket?"
)


# -------------------------------------------------
# Inspect The Raw ToolCall
# -------------------------------------------------

print("\n" + "=" * 60)
print("MODEL RESPONSE")
print("=" * 60)

print(f"Content:    {response.content!r}")
print(f"Tool Calls: {response.tool_calls}")

for tool_call in response.tool_calls:

    print(f"\n  name: {tool_call['name']}")
    print(f"  args: {tool_call['args']}")
    print(f"  id:   {tool_call['id']}")
