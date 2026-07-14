"""
Lesson 05 - Understanding Chunk Overlap

Goal:
- Learn why chunk_overlap exists.
- Compare overlap vs no overlap.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------
# Sample Text
# ------------------------------------------------

text = """
Artificial Intelligence is transforming industries.

Machine Learning is a subset of AI.

Deep Learning uses neural networks.

Natural Language Processing helps computers understand text.

Computer Vision allows machines to understand images.

LangChain makes building LLM applications easier.
"""

# ------------------------------------------------
# Without Overlap
# ------------------------------------------------

print("=" * 60)
print("WITHOUT OVERLAP")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=0
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, start=1):

    print("\n")
    print("=" * 40)
    print(f"Chunk {i}")
    print("=" * 40)

    print(chunk)

# ------------------------------------------------
# With Overlap
# ------------------------------------------------

print("\n\n")
print("=" * 60)
print("WITH OVERLAP")
print("=" * 60)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=80,
    chunk_overlap=20
)

chunks = splitter.split_text(text)

for i, chunk in enumerate(chunks, start=1):

    print("\n")
    print("=" * 40)
    print(f"Chunk {i}")
    print("=" * 40)

    print(chunk)