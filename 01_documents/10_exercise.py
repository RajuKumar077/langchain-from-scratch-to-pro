"""
Lesson 10 - Final Exercise

Goal:
- Load multiple files
- Work with Document objects
- Read page_content
- Read metadata
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

# ==========================================================
# Folder
# ==========================================================

BASE_DIR = Path(__file__).parent.parent

DATA_PATH = BASE_DIR / "data" / "documents"

# ==========================================================
# Load Documents
# ==========================================================

loader = DirectoryLoader(

    path=DATA_PATH,

    glob="*.txt",

    loader_cls=TextLoader

)

documents = loader.load()

# ==========================================================
# Basic Information
# ==========================================================

print("=" * 60)

print("Total Documents")

print("=" * 60)

print(len(documents))

# ==========================================================
# Loop Through Documents
# ==========================================================

for index, document in enumerate(documents, start=1):

    print("\n" + "=" * 60)

    print(f"Document {index}")

    print("=" * 60)

    print("Content")

    print(document.page_content)

    print("\nMetadata")

    print(document.metadata)

# ==========================================================
# First Document
# ==========================================================

print("\n" + "=" * 60)

print("First Document")

print("=" * 60)

print(documents[0])

# ==========================================================
# First Document Content
# ==========================================================

print("\nContent")

print(documents[0].page_content)

# ==========================================================
# First Document Metadata
# ==========================================================

print("\nMetadata")

print(documents[0].metadata)

# ==========================================================
# Count Total Characters
# ==========================================================

total_characters = 0

for document in documents:

    total_characters += len(document.page_content)

print("\n" + "=" * 60)

print("Total Characters")

print("=" * 60)

print(total_characters)

# ==========================================================
# Print All Sources
# ==========================================================

print("\n" + "=" * 60)

print("Source Files")

print("=" * 60)

for document in documents:

    print(document.metadata["source"])