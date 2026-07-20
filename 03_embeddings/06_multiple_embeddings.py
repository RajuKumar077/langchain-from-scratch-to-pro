"""
Lesson 06 - Embedding Multiple Documents

Goal
----
- Load a text file
- Split it into chunks
- Generate embeddings for every chunk
"""

from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader

# -------------------------------------------------
# Locate the sample file
# -------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "sample.txt"

# -------------------------------------------------
# Load Document
# -------------------------------------------------

loader = TextLoader(str(DATA_FILE))
documents = loader.load()

print("=" * 60)
print("Loaded Documents")
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

print("\n" + "=" * 60)
print("Chunks Created")
print("=" * 60)

print(len(chunks))

# -------------------------------------------------
# Load Embedding Model
# -------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("\n✅ Embedding Model Loaded")

# -------------------------------------------------
# Embed Every Chunk
# -------------------------------------------------

texts = [chunk.page_content for chunk in chunks]

vectors = embeddings.embed_documents(texts)

print("\n" + "=" * 60)
print("Embeddings Created")
print("=" * 60)

print(len(vectors))

# -------------------------------------------------
# Display Results
# -------------------------------------------------

for i in range(len(chunks)):

    print("\n" + "=" * 60)
    print(f"Chunk {i+1}")
    print("=" * 60)

    print("\nChunk Text:")
    print(chunks[i].page_content)

    print("\nEmbedding Length:")
    print(len(vectors[i]))

    print("\nFirst 5 Numbers:")
    print(vectors[i][:5])