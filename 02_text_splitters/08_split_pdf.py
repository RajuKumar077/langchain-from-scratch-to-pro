"""
Lesson 08 - split_documents()

Goal:
- Split Document objects.
- Metadata is preserved.
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------
# Load document
# ---------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

loader = TextLoader(ROOT / "data" / "sample.txt")

documents = loader.load()

print("=" * 60)
print("Before Splitting")
print("=" * 60)

print(f"Documents : {len(documents)}")

# ---------------------------------------------------
# Create splitter
# ---------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=120,
    chunk_overlap=30
)

# ---------------------------------------------------
# Split Documents
# ---------------------------------------------------

chunks = splitter.split_documents(documents)

print("\n" + "=" * 60)
print("After Splitting")
print("=" * 60)

print(f"Chunks : {len(chunks)}")

# ---------------------------------------------------
# Display chunks
# ---------------------------------------------------

for i, chunk in enumerate(chunks, start=1):

    print("\n" + "=" * 60)
    print(f"Chunk {i}")
    print("=" * 60)

    print("Metadata")

    print(chunk.metadata)

    print("\nContent")

    print(chunk.page_content)