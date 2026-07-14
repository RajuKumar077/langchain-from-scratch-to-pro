"""
Lesson 06 - Document IDs

Goal:
- Create unique IDs
- Attach IDs to Documents
- Understand why IDs are useful
"""

import uuid

from langchain_core.documents import Document

# ==========================================================
# Generate Unique IDs
# ==========================================================

id1 = str(uuid.uuid4())
id2 = str(uuid.uuid4())
id3 = str(uuid.uuid4())


print("=" * 60)
print("Generated IDs")
print("=" * 60)

print(id1)
print(id2)
print(id3)

# ==========================================================
# Create Documents with IDs
# ==========================================================

documents = [

    Document(
        id=id1,
        page_content="Artificial Intelligence",
        metadata={
            "topic": "AI"
        }
    ),

    Document(
        id=id2,
        page_content="Machine Learning",
        metadata={
            "topic": "ML"
        }
    ),

    Document(
        id=id3,
        page_content="Deep Learning",
        metadata={
            "topic": "DL"
        }
    ),

    Document(
        id=str(uuid.uuid4()),
        page_content="Large Language Models power ChatGPT.",
        metadata={
            "topic": "LLM"
        }
    )

]

# ==========================================================
# Print Documents
# ==========================================================

print("\n" + "=" * 60)
print("Documents")
print("=" * 60)

for document in documents:

    print("\nDocument ID :", document.id)
    print("Content     :", document.page_content)
    print("Metadata    :", document.metadata)

# ==========================================================
# Access One Document
# ==========================================================

print("\n" + "=" * 60)
print("First Document")
print("=" * 60)

print(documents[0])

# ==========================================================
# Print Only IDs
# ==========================================================

print("\n" + "=" * 60)
print("All IDs")
print("=" * 60)

for document in documents:
    print(document.id)