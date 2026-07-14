"""
Lesson 03 - Creating a Document Manually

Goal:
- Create a LangChain Document manually
- Access page_content and metadata
- Update metadata
"""

from langchain_core.documents import Document

# ==========================================================
# Create a Document
# ==========================================================

document = Document(
    page_content="""
Artificial Intelligence is transforming the world.
Machine Learning is a subset of AI.
Deep Learning is a subset of Machine Learning.
""",
    metadata={
        "source": "sample.txt",
        "author": "Raju Kumar",
        "category": "AI Notes",
        "language": "English",
        "chapter": 1,
    },
)

# ==========================================================
# Complete Document
# ==========================================================

print("\n" + "=" * 60)
print("Complete Document")
print("=" * 60)

print(document)

# ==========================================================
# Page Content
# ==========================================================

print("\n" + "=" * 60)
print("Page Content")
print("=" * 60)

print(document.page_content)

# ==========================================================
# Metadata
# ==========================================================

print("\n" + "=" * 60)
print("Metadata")
print("=" * 60)

print(document.metadata)

# ==========================================================
# Read Individual Metadata
# ==========================================================

print("\n" + "=" * 60)
print("Individual Metadata")
print("=" * 60)

print("Source   :", document.metadata["source"])
print("Author   :", document.metadata["author"])
print("Language :", document.metadata["language"])
print("Chapter  :", document.metadata["chapter"])

# ==========================================================
# Add New Metadata
# ==========================================================

document.metadata["difficulty"] = "Beginner"
document.metadata["framework"] = "LangChain"

print("\n" + "=" * 60)
print("Updated Metadata")
print("=" * 60)

print(document.metadata)