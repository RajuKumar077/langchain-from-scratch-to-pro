"""
Lesson 06 - Visualizing Chunks

Goal:
- Visualize how one document becomes many documents.
- Understand split_documents().
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# -------------------------------------------------------
# Load Document
# -------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

loader = TextLoader(ROOT / "data" / "sample.txt")

documents = loader.load()

print("=" * 70)
print("Original Document")
print("=" * 70)

print(f"Total Documents : {len(documents)}")

print("\nMetadata")
print(documents[0].metadata)

print("\nCharacters")
print(len(documents[0].page_content))

# -------------------------------------------------------
# Split Document
# -------------------------------------------------------

splitter = RecursiveCharacterTextSplitter(

    chunk_size=120,
    chunk_overlap=30

)

chunks = splitter.split_documents(documents)

# -------------------------------------------------------
# Results
# -------------------------------------------------------

print("\n")
print("=" * 70)
print("After Chunking")
print("=" * 70)

print(f"Original Documents : {len(documents)}")
print(f"Chunks Created     : {len(chunks)}")

# -------------------------------------------------------
# Show every chunk
# -------------------------------------------------------

for i, chunk in enumerate(chunks, start=1):

    print("\n")
    print("=" * 70)
    print(f"Chunk {i}")
    print("=" * 70)

    print("Characters :", len(chunk.page_content))

    print("\nMetadata")

    print(chunk.metadata)

    print("\nContent")

    print(chunk.page_content)