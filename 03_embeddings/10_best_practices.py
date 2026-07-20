"""
Lesson 10 - Similarity Search with Scores

Goal
----
- Retrieve documents
- See similarity scores
"""

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

# -----------------------------------------
# PostgreSQL Connection
# -----------------------------------------

CONNECTION = (
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)

COLLECTION_NAME = "lesson08"

# -----------------------------------------
# Embedding Model
# -----------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

# -----------------------------------------
# Connect to PGVector
# -----------------------------------------

vector_store = PGVector(
    embeddings=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)

query = "Explain Machine Learning"

# -----------------------------------------
# Search with Scores
# -----------------------------------------

results = vector_store.similarity_search_with_score(
    query,
    k=3
)

# -----------------------------------------
# Display Results
# -----------------------------------------

print("=" * 60)
print("Similarity Search with Scores")
print("=" * 60)

for i, (doc, score) in enumerate(results, start=1):

    print(f"\nResult {i}")

    print("-" * 50)

    print("Score:")
    print(score)

    print("\nText:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)