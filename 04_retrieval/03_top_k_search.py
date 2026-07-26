"""
Lesson 03 - Top K Search

Goal
----
- Understand the importance of K
- Retrieve different numbers of documents
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
print("Top K Retrieval Demo")
print("=" * 60)

query = input("\nAsk a Question: ")

# -------------------------------------------------
# Ask user for K
# -------------------------------------------------

k = int(input("How many documents should be retrieved? "))

# -------------------------------------------------
# Search
# -------------------------------------------------

results = vector_store.similarity_search(
    query=query,
    k=k
)

# -------------------------------------------------
# Display
# -------------------------------------------------

print("\n" + "=" * 60)
print(f"Top {k} Documents")
print("=" * 60)

for i, doc in enumerate(results, start=1):

    print(f"\nResult {i}")

    print("-" * 50)

    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)