"""
Lesson 09 - Similarity Search

Goal
----
- Ask a question
- Convert it into an embedding
- Retrieve the most similar chunks
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
# Connect to Existing Vector Store
# -----------------------------------------

vector_store = PGVector(
    embeddings=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)

print("=" * 60)
print("Connected to PGVector")
print("=" * 60)

# -----------------------------------------
# User Query
# -----------------------------------------

query = "What is Machine Learning?"

print("\nQuestion:")
print(query)

# -----------------------------------------
# Similarity Search
# -----------------------------------------

results = vector_store.similarity_search(
    query,
    k=3
)

# -----------------------------------------
# Display Results
# -----------------------------------------

print("\n" + "=" * 60)
print("Top Matching Chunks")
print("=" * 60)

for i, doc in enumerate(results, start=1):

    print(f"\nResult {i}")

    print("-" * 40)

    print("Text:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)