from pathlib import Path

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter


# ============================================================
# Configuration
# ============================================================

POSTGRES_URL = (
    "postgresql+psycopg://langchain:langchain"
    "@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# File to add
# ============================================================

FILE_PATH = "data/sample.txt"


# ============================================================
# Load file
# ============================================================

print(f"Loading: {FILE_PATH}")

text = Path(FILE_PATH).read_text(encoding="utf-8")

document = Document(
    page_content=text,
    metadata={
        "source": FILE_PATH
    }
)

print(f"Loaded {len(text)} characters")


# ============================================================
# Split into chunks
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

chunks = splitter.split_documents([document])

print(f"Created {len(chunks)} chunks")


# ============================================================
# Embedding model
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# Connect to existing PGVector collection
# ============================================================

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=POSTGRES_URL,
    use_jsonb=True,
)


# ============================================================
# Add new documents
# ============================================================

vector_store.add_documents(chunks)

print()
print("=" * 60)
print("SUCCESS")
print("=" * 60)

print(f"Added {len(chunks)} chunks")
print(f"Source: {FILE_PATH}")
print(f"Collection: {COLLECTION_NAME}")