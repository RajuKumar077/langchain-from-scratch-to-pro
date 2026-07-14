"""
Lesson 02 - Understanding the Document Object

Goal:
- Access page_content
- Access metadata
- Check object type
"""

from pathlib import Path

from langchain_community.document_loaders import TextLoader

# --------------------------------------------------
# Locate the data file
# --------------------------------------------------

BASE_DIR = Path(__file__).parent.parent
FILE_PATH = BASE_DIR / "data" / "sample.txt"

# --------------------------------------------------
# Load the document
# --------------------------------------------------

loader = TextLoader(str(FILE_PATH))
documents = loader.load()

# Get the first Document
doc = documents[0]

print("=" * 60)
print("Document Object")
print("=" * 60)

print(doc)

print("\n")

# --------------------------------------------------
# Access the actual text
# --------------------------------------------------

print("=" * 60)
print("Page Content")
print("=" * 60)

print(doc.page_content)

print("\n")

# --------------------------------------------------
# Access metadata
# --------------------------------------------------

print("=" * 60)
print("Metadata")
print("=" * 60)

print(doc.metadata)

print("\n")

# --------------------------------------------------
# Object Type
# --------------------------------------------------

print("=" * 60)
print("Object Type")
print("=" * 60)

print(type(doc))