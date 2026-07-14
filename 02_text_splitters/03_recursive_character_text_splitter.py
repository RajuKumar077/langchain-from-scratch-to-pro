"""
Lesson 03 - RecursiveCharacterTextSplitter

Goal:
- Learn the most commonly used text splitter.
- Understand how it creates smarter chunks.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ------------------------------------------------
# Sample Text
# ------------------------------------------------

text = """
Artificial Intelligence is transforming industries.

Machine Learning helps computers learn from data.

Deep Learning uses neural networks.

Natural Language Processing understands text.

Computer Vision understands images.

LangChain helps developers build LLM applications.

Vector Databases store embeddings.

RAG combines retrieval with language models.
"""

print("=" * 60)
print("Original Text")
print("=" * 60)

print(text)

# ------------------------------------------------
# Create Recursive Splitter
# ------------------------------------------------

splitter = RecursiveCharacterTextSplitter(

    chunk_size=80,

    chunk_overlap=0

)

# ------------------------------------------------
# Split
# ------------------------------------------------

chunks = splitter.split_text(text)

# ------------------------------------------------
# Results
# ------------------------------------------------

print("\n")
print("=" * 60)
print(f"Total Chunks : {len(chunks)}")
print("=" * 60)

for i, chunk in enumerate(chunks, start=1):

    print("\n" + "=" * 60)
    print(f"Chunk {i}")
    print("=" * 60)

    print(chunk)

    print("\nCharacters :", len(chunk))