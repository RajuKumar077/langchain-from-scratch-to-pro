"""
Lesson 08 - Store Embeddings in PGVector

Goal
----
- Load a document
- Split it into chunks
- Generate embeddings
- Store everything inside PostgreSQL + PGVector
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

COLLECTION_NAME = "lesson08"

# -------------------------------------------------
# Locate File
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "sample.txt"

# -------------------------------------------------
# Load Document
# -------------------------------------------------

loader = TextLoader(str(DATA_FILE))
documents = loader.load()

print("=" * 60)
print("Documents Loaded")
print("=" * 60)

print(len(documents))

# -------------------------------------------------
# Split into Chunks
# -------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

chunks = splitter.split_documents(documents)

print("\nChunks Created:", len(chunks))

# -------------------------------------------------
# Embedding Model
# -------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("Embedding Model Ready")

# -------------------------------------------------
# Store into PGVector
# -------------------------------------------------

vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)

print("\nEmbeddings Stored Successfully!")

print(f"Collection : {COLLECTION_NAME}")
print(f"Chunks     : {len(chunks)}")