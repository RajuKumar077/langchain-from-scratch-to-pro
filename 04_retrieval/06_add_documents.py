"""
Lesson 06 - Add Documents to Existing PGVector Collection

Goal
----
- Connect to an existing collection
- Add new documents
- Verify they were stored
"""

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

# -------------------------------------------------
# PostgreSQL Connection
# -------------------------------------------------

CONNECTION = (
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)

COLLECTION_NAME = "lesson08"

# -------------------------------------------------
# Embedding Model
# -------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# -------------------------------------------------
# Connect to Existing Collection
# -------------------------------------------------

vector_store = PGVector(
    embeddings=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)

print("=" * 60)
print("Connected to Existing Collection")
print("=" * 60)

# -------------------------------------------------
# Create New Documents
# -------------------------------------------------

new_documents = [

    Document(
        page_content="LangGraph is used for building stateful AI agents.",
        metadata={
            "source": "langgraph.txt",
            "topic": "LangGraph"
        }
    ),

    Document(
        page_content="Vector databases store embeddings for semantic search.",
        metadata={
            "source": "vectordb.txt",
            "topic": "Vector Database"
        }
    )

]

# -------------------------------------------------
# Add Documents
# -------------------------------------------------

ids = vector_store.add_documents(new_documents)

print("\nDocuments Added Successfully!")

print("\nGenerated IDs:")

for doc_id in ids:
    print(doc_id)

# -------------------------------------------------
# Verify
# -------------------------------------------------

print("\n" + "=" * 60)
print("Verification Search")
print("=" * 60)

results = vector_store.similarity_search(
    "LangGraph",
    k=2
)

for i, doc in enumerate(results, start=1):

    print(f"\nResult {i}")

    print("-" * 50)

    print(doc.page_content)

    print("\nMetadata:")

    print(doc.metadata)