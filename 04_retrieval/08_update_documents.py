"""
Lesson 08 - Update Documents in PGVector

Goal
----
- Delete an old document
- Add the updated version
- Verify the update
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
# Connect to Collection
# -------------------------------------------------

vector_store = PGVector(
    embeddings=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)

print("=" * 60)
print("Update Document Demo")
print("=" * 60)

# -------------------------------------------------
# Document ID to Update
# -------------------------------------------------

document_id = "langgraph_001"

# -------------------------------------------------
# Delete Old Document
# -------------------------------------------------

vector_store.delete(
    ids=[document_id]
)

print(f"\nDeleted: {document_id}")

# -------------------------------------------------
# Create Updated Document
# -------------------------------------------------

updated_document = Document(
    page_content="""
LangGraph is a framework for building
stateful, multi-agent AI applications
using graphs and workflows.
""",
    metadata={
        "source": "langgraph.txt",
        "topic": "LangGraph"
    }
)

# -------------------------------------------------
# Add Updated Document
# -------------------------------------------------

vector_store.add_documents(
    documents=[updated_document],
    ids=[document_id]
)

print("Updated document added.")

# -------------------------------------------------
# Verify Update
# -------------------------------------------------

results = vector_store.similarity_search(
    query="LangGraph",
    k=1
)

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

for doc in results:

    print("\nContent:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)