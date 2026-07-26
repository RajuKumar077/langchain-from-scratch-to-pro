"""
Lesson 02 - Similarity Search with Scores

Goal
----
- Retrieve similar documents
- View their similarity scores
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
print("Connected to PGVector")
print("=" * 60)

# -------------------------------------------------
# Ask User
# -------------------------------------------------

query = input("\nAsk a Question: ")

# -------------------------------------------------
# Search with Scores
# -------------------------------------------------

results = vector_store.similarity_search_with_score(
    query=query,
    k=3
)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("\n" + "=" * 60)
print("Top 3 Results")
print("=" * 60)

for i, (doc, score) in enumerate(results, start=1):

    print(f"\nResult {i}")
    print("-" * 40)

    print(f"Score : {score}")

    print("\nContent:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)