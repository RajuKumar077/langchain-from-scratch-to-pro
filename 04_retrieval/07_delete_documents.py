"""
Lesson 07 - Delete Documents from PGVector

Goal
----
- Search documents
- Get their IDs
- Delete documents
"""

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
print("Delete Documents Demo")
print("=" * 60)

# -------------------------------------------------
# Search Documents
# -------------------------------------------------

results = vector_store.similarity_search_with_score(
    query="LangGraph",
    k=2
)

print("\nDocuments Found")

document_ids = []

for i, (doc, score) in enumerate(results, start=1):

    print("\n" + "-" * 50)
    print(f"Result {i}")

    print("\nContent:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)

    # Get document id
    doc_id = doc.metadata.get("id")

    print("\nID:")
    print(doc_id)

    if doc_id:
        document_ids.append(doc_id)

# -------------------------------------------------
# Delete Documents
# -------------------------------------------------

if document_ids:

    vector_store.delete(ids=document_ids)

    print("\nDocuments Deleted Successfully!")

else:

    print("\nNo IDs found. Nothing deleted.")