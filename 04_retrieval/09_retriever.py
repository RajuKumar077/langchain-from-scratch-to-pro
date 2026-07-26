"""
Lesson 09 - Using a Retriever

Goal
----
- Convert a Vector Store into a Retriever
- Retrieve documents using invoke()
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

# -------------------------------------------------
# Create Retriever
# -------------------------------------------------

retriever = vector_store.as_retriever()

print("=" * 60)
print("Retriever Created")
print("=" * 60)

# -------------------------------------------------
# Ask Question
# -------------------------------------------------

query = input("\nAsk a Question: ")

# -------------------------------------------------
# Retrieve Documents
# -------------------------------------------------

documents = retriever.invoke(query)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("\n" + "=" * 60)
print("Retrieved Documents")
print("=" * 60)

for i, doc in enumerate(documents, start=1):

    print(f"\nResult {i}")
    print("-" * 50)

    print("Content:")
    print(doc.page_content)

    print("\nMetadata:")
    print(doc.metadata)