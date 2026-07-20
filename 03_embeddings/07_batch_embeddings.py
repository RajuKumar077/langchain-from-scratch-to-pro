"""
Lesson 07 - Batch Embeddings

Goal
----
- Learn why batch embedding is faster
- Compare one-by-one embedding vs batch embedding
"""

from pathlib import Path
import time

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings

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

# -------------------------------------------------
# Split into chunks
# -------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20,
)

chunks = splitter.split_documents(documents)

print("=" * 60)
print("Total Chunks")
print("=" * 60)

print(len(chunks))

# -------------------------------------------------
# Load Embedding Model
# -------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("\n✅ Model Loaded")

# -------------------------------------------------
# Method 1 - One by One
# -------------------------------------------------

print("\n" + "=" * 60)
print("Embedding One by One")
print("=" * 60)

start = time.time()

vectors1 = []

for chunk in chunks:
    vector = embeddings.embed_query(chunk.page_content)
    vectors1.append(vector)

end = time.time()

print(f"Generated {len(vectors1)} embeddings")
print(f"Time Taken : {end-start:.2f} seconds")

# -------------------------------------------------
# Method 2 - Batch Embedding
# -------------------------------------------------

print("\n" + "=" * 60)
print("Batch Embedding")
print("=" * 60)

texts = [chunk.page_content for chunk in chunks]

start = time.time()

vectors2 = embeddings.embed_documents(texts)

end = time.time()

print(f"Generated {len(vectors2)} embeddings")
print(f"Time Taken : {end-start:.2f} seconds")

# -------------------------------------------------
# Verify
# -------------------------------------------------

print("\n" + "=" * 60)
print("Verification")
print("=" * 60)

print("One by One :", len(vectors1))
print("Batch      :", len(vectors2))

print("\nBoth methods generated the same number of embeddings.")