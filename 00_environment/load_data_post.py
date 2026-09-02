from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_text_splitters import RecursiveCharacterTextSplitter


POSTGRES_URL = (
    "postgresql+psycopg://langchain:langchain"
    "@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"
EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# 1. Load text file
# ============================================================

file_path = Path("data/sample.txt")

text = file_path.read_text(encoding="utf-8")

print(f"Loaded {len(text)} characters")


# ============================================================
# 2. Convert text into chunks
# ============================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
)

documents = splitter.create_documents([text])

print(f"Created {len(documents)} chunks")


# ============================================================
# 3. Create embeddings
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# 4. Connect to PGVector
# ============================================================

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=POSTGRES_URL,
    use_jsonb=True,
)


# ============================================================
# 5. Store chunks + embeddings
# ============================================================

vector_store.add_documents(documents)

print("✓ Documents successfully stored in PGVector")