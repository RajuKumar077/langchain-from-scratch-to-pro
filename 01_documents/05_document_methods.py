"""
Lesson 04 - Working with Multiple Documents

Goal:
- Create multiple Document objects
- Store them in a list
- Access each Document
"""

from langchain_core.documents import Document

# ==========================================================
# Create Multiple Documents
# ==========================================================

documents = [

    Document(
        page_content="Artificial Intelligence is transforming the world.",
        metadata={
            "topic": "Artificial Intelligence",
            "chapter": 1
        }
    ),

    Document(
        page_content="Machine Learning is a subset of Artificial Intelligence.",
        metadata={
            "topic": "Machine Learning",
            "chapter": 2
        }
    ),

    Document(
        page_content="Deep Learning is a subset of Machine Learning.",
        metadata={
            "topic": "Deep Learning",
            "chapter": 3
        }
    ),

]

# ==========================================================
# Total Documents
# ==========================================================

print("=" * 60)
print("Total Documents")
print("=" * 60)

print(len(documents))

# ==========================================================
# Print Every Document
# ==========================================================

print("\n" + "=" * 60)
print("All Documents")
print("=" * 60)

for index, document in enumerate(documents, start=1):

    print(f"\nDocument {index}")
    print("-" * 40)

    print("Page Content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)

# ==========================================================
# Access Individual Documents
# ==========================================================

print("\n" + "=" * 60)
print("Access Individual Documents")
print("=" * 60)

print("First Topic :", documents[0].metadata["topic"])
print("Second Topic:", documents[1].metadata["topic"])
print("Third Topic :", documents[2].metadata["topic"])

# ==========================================================
# Loop Through Only Page Content
# ==========================================================

print("\n" + "=" * 60)
print("Only Page Content")
print("=" * 60)

for document in documents:
    print(document.page_content)

# ==========================================================
# Loop Through Only Metadata
# ==========================================================

print("\n" + "=" * 60)
print("Only Metadata")
print("=" * 60)

for document in documents:
    print(document.metadata)