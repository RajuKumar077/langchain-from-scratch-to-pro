"""
Lesson 04 - Understanding Chunk Size

Goal:
- See how chunk_size changes the number of chunks.
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

Vector databases store embeddings.

RAG combines retrieval with language models.
"""

# ------------------------------------------------
# Different chunk sizes
# ------------------------------------------------

sizes = [50, 100, 200]

# ------------------------------------------------
# Test each size
# ------------------------------------------------

for size in sizes:

    print("\n")
    print("=" * 60)
    print(f"Chunk Size = {size}")
    print("=" * 60)

    splitter = RecursiveCharacterTextSplitter(

        chunk_size=size,

        chunk_overlap=0

    )

    chunks = splitter.split_text(text)

    print("Total Chunks:", len(chunks))

    for i, chunk in enumerate(chunks, start=1):

        print(f"\nChunk {i}")
        print(chunk)
        print("Characters:", len(chunk))