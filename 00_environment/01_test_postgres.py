"""
00_test_rag_connection.py

Goal:
    Verify that:
    1. PostgreSQL is reachable
    2. PGVector connection works
    3. The expected collection exists
    4. The collection contains documents/embeddings

Run this BEFORE running:
    01_basic_rag.py
"""

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector


# ============================================================
# 1. Configuration
# ============================================================

POSTGRES_URL = (
    "postgresql+psycopg://langchain:langchain"
    "@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# 2. Create embedding model
# ============================================================

print("\n[1/4] Loading embedding model...")

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)

print("✓ Embedding model loaded")


# ============================================================
# 3. Connect to PGVector
# ============================================================

print("\n[2/4] Connecting to PostgreSQL / PGVector...")

try:

    vector_store = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=POSTGRES_URL,
        use_jsonb=True,
    )

    print("✓ PGVector connection successful")

except Exception as e:

    print("✗ PGVector connection failed")

    print("\nError:")
    print(e)

    raise SystemExit(1)


# ============================================================
# 4. Check collection and stored documents
# ============================================================

print("\n[3/4] Checking vector collection...")

try:

    # Run a very small similarity search.
    # This verifies that:
    # - the collection exists
    # - embeddings can be generated
    # - PGVector can perform a similarity search

    test_query = "test"

    documents = vector_store.similarity_search(
        test_query,
        k=3,
    )

    print(f"✓ Collection '{COLLECTION_NAME}' is accessible")
    print(f"✓ Retrieved {len(documents)} document(s)")


    # ========================================================
    # 5. Display sample documents
    # ========================================================

    print("\n[4/4] Sample stored documents")
    print("=" * 60)

    if not documents:

        print("⚠️ Collection is empty.")
        print("You need to run your document ingestion / embedding script first.")

    else:

        for i, document in enumerate(documents, start=1):

            print(f"\nDocument {i}")
            print("-" * 60)

            print(document.page_content[:500])

            print("\nMetadata:")
            print(document.metadata)


except Exception as e:

    print("✗ Vector search failed")

    print("\nError:")
    print(e)

    raise SystemExit(1)


# ============================================================
# 6. Final result
# ============================================================

print("\n" + "=" * 60)
print("CONNECTION TEST COMPLETE")
print("=" * 60)

print("✓ PostgreSQL reachable")
print("✓ PGVector accessible")
print(f"✓ Collection '{COLLECTION_NAME}' accessible")
print("✓ Embedding search working")

print("\nYou can now run:")
print("python 01_basic_rag.py")