"""
Lesson 09 - Directory Loader

Goal:
- Load every text file from a folder
- Understand that every file becomes one Document
"""

from pathlib import Path

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import TextLoader

# ==========================================================
# Folder Path
# ==========================================================

BASE_DIR = Path(__file__).parent.parent
DATA_PATH = BASE_DIR / "data" / "documents"

# ==========================================================
# Create Loader
# ==========================================================

loader = DirectoryLoader(
    path=DATA_PATH,
    glob="*.txt",
    loader_cls=TextLoader
)

print("=" * 60)
print("Directory Loader Created")
print("=" * 60)

# ==========================================================
# Load Documents
# ==========================================================

documents = loader.load()

print("\nTotal Documents:", len(documents))

# ==========================================================
# Print Documents
# ==========================================================

for index, document in enumerate(documents, start=1):

    print("\n" + "=" * 60)
    print(f"Document {index}")
    print("=" * 60)

    print(document.page_content)

    print("\nMetadata")

    print(document.metadata)