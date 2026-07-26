"""
Lesson 10 - Complete Retrieval Pipeline

Goal
----
Complete flow:
Text File
    ↓
Document Loader
    ↓
Chunking
    ↓
Embeddings
    ↓
PGVector
    ↓
Retriever
    ↓
Relevant Documents
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector

# -------------------------------------------------
# PostgreSQL Connection
# -------------------------------------------------

CONNECTION = (
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"

# -------------------------------------------------
# Locate File
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "sample.txt"

# -------------------------------------------------
# Load Documents
# -------------------------------------------------

loader = TextLoader(str(DATA_FILE))
documents = loader.load()

print("=" * 60)
print("Step 1 - Documents Loaded")
print("=" * 60)

print(f"Documents : {len(documents)}")

# -------------------------------------------------
# Split Documents
# -------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

chunks = splitter.split_documents(documents)

print("\nStep 2 - Chunking Complete")

print(f"Chunks : {len(chunks)}")

# -------------------------------------------------
# Embedding Model
# -------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("\nStep 3 - Embedding Model Ready")

# -------------------------------------------------
# Store in PGVector
# -------------------------------------------------

vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)

print("\nStep 4 - Embeddings Stored")

# -------------------------------------------------
# Create Retriever
# -------------------------------------------------

retriever = vector_store.as_retriever()

print("\nStep 5 - Retriever Created")

# -------------------------------------------------
# Ask Question
# -------------------------------------------------

query = input("\nAsk a Question: ")

# -------------------------------------------------
# Retrieve Documents
# -------------------------------------------------

results = retriever.invoke(query)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("\n" + "=" * 60)
print("Retrieved Documents")
print("=" * 60)

for i, doc in enumerate(results, start=1):

    print(f"\nResult {i}")

    print("-" * 50)

    print(doc.page_content)

    print("\nMetadata:")

    print(doc.metadata)

print("\nPipeline Completed Successfully!")