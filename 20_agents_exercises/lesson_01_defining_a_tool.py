"""
Module 20 - Agent Architecture
Lesson 01 - Defining a @tool

Goal
----
- Wrap the existing PGVector retriever as a LangChain tool
- Understand that the docstring, not the code, is what the LLM reads
"""

from langchain_core.tools import tool
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector


# -------------------------------------------------
# PostgreSQL Connection (same as Module 05 RAG lesson)
# -------------------------------------------------

CONNECTION = (
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"


# -------------------------------------------------
# Embedding Model
# -------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
)


# -------------------------------------------------
# Connect To PGVector
# -------------------------------------------------

vector_store = PGVector(
    embeddings=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)


# -------------------------------------------------
# Create Retriever
# -------------------------------------------------

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
# Inspect What The LLM Actually Sees
# -------------------------------------------------

print("\n" + "=" * 60)
print("TOOL SCHEMA (WHAT THE LLM SEES)")
print("=" * 60)

print(f"Name:        {search_knowledge_base.name}")
print(f"Description: {search_knowledge_base.description}")
print(f"Args Schema: {search_knowledge_base.args}")


# -------------------------------------------------
# Call The Tool Directly (No LLM Involved Yet)
# -------------------------------------------------

result = search_knowledge_base.invoke({
    "query": "duplicate invoice charge",
})

print("\n" + "=" * 60)
print("DIRECT TOOL CALL RESULT")
print("=" * 60)

print(result)
