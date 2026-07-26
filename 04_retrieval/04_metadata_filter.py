"""
Lesson 04 - Metadata Filtering

Goal
----
- Understand metadata filtering
- Search only matching documents
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
# Connect to PGVector
# -------------------------------------------------

vector_store = PGVector(
    embeddings=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)

print("=" * 60)
print("Metadata Filter Demo")
print("=" * 60)

# -------------------------------------------------
# Show Stored Metadata
# -------------------------------------------------

print("\nChecking Stored Metadata...\n")

docs = vector_store.similarity_search(
    query="AI",
    k=1
)

if docs:
    print("Stored Metadata:")
    print(docs[0].metadata)
else:
    print("No documents found!")
    exit()

# -------------------------------------------------
# Ask User
# -------------------------------------------------

query = input("\nAsk a Question: ")

# -------------------------------------------------
# Use EXACT metadata value
# -------------------------------------------------

metadata_value = docs[0].metadata["source"]

results = vector_store.similarity_search(
    query=query,
    k=3,
    filter={
        "source": metadata_value
    }
)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("\n" + "=" * 60)
print("Filtered Results")
print("=" * 60)

if not results:
    print("No matching documents found.")
else:
    for i, doc in enumerate(results, start=1):

        print(f"\nResult {i}")
        print("-" * 50)

        print("Content:")
        print(doc.page_content)

        print("\nMetadata:")
        print(doc.metadata)