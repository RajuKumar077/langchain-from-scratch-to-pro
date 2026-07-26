"""
Lesson 01 - Basic Similarity Search

Goal
----
- Connect to an existing PGVector collection
- Ask a question
- Retrieve the most relevant documents
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
# Load Embedding Model
# -------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("=" * 60)
print("Embedding Model Loaded")
print("=" * 60)

# -------------------------------------------------
# Connect to Existing Vector Store
# -------------------------------------------------

vector_store = PGVector(
    embeddings=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)

print("\nConnected to PGVector!")

# -------------------------------------------------
# Ask a Question
# -------------------------------------------------

query = input("\nAsk a Question: ")

# -------------------------------------------------
# Retrieve Similar Documents
# -------------------------------------------------

results = vector_store.similarity_search(
    query=query,
    k=3
)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("\n" + "=" * 60)
print("Top 3 Matching Documents")
print("=" * 60)

for i, doc in enumerate(results, start=1):

    print(f"\nResult {i}")

    print("-" * 40)

    print("Page Content:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)