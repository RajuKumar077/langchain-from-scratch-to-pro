"""
Lesson 09 - Splitting CSV Documents

Goal:
- Load a CSV file.
- Split each row into chunks.
"""

from pathlib import Path

from langchain_community.document_loaders import CSVLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

ROOT = Path(__file__).resolve().parent.parent

loader = CSVLoader(ROOT / "data" / "employees.csv")

documents = loader.load()

print("=" * 60)
print("Rows Loaded")
print("=" * 60)

print(len(documents))

splitter = RecursiveCharacterTextSplitter(
    chunk_size=40,
    chunk_overlap=10
)

chunks = splitter.split_documents(documents)

print("\n" + "=" * 60)
print("Chunks Created")
print("=" * 60)

print(len(chunks))

for i, chunk in enumerate(chunks, start=1):

    print("\n" + "=" * 60)
    print(f"Chunk {i}")
    print("=" * 60)

    print("Metadata")

    print(chunk.metadata)

    print("\nContent")

    print(chunk.page_content)