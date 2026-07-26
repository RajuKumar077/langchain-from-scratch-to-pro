"""
Lesson 05 - Dynamic Metadata Search

Goal
----
- Let the user choose the metadata filter
- Perform similarity search using that filter
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
print("Dynamic Metadata Search")
print("=" * 60)

# -------------------------------------------------
# Show Available Metadata
# -------------------------------------------------

sample = vector_store.similarity_search(
    query="AI",
    k=1
)

if not sample:
    print("No documents found!")
    exit()

print("\nAvailable Metadata:")
print(sample[0].metadata)

# -------------------------------------------------
# User Input
# -------------------------------------------------

query = input("\nAsk a Question: ")

metadata_key = input("Metadata Key (example: source): ")

metadata_value = input("Metadata Value: ")

# -------------------------------------------------
# Search
# -------------------------------------------------

results = vector_store.similarity_search(
    query=query,
    k=3,
    filter={
        metadata_key: metadata_value
    }
)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("\n" + "=" * 60)
print("Results")
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