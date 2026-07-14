"""
Lesson 07 - split_text()

Goal:
- Split a normal Python string.
- No Document objects are involved.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

# ---------------------------------------------------
# A normal Python string
# ---------------------------------------------------

text = """
Artificial Intelligence is transforming the world.

Machine Learning is a subset of AI.

Deep Learning is a subset of Machine Learning.

Large Language Models are trained on huge datasets.

LangChain helps build LLM applications.

Vector Databases store embeddings.

Ollama runs models locally.

PostgreSQL + PGVector stores vectors efficiently.
"""

# ---------------------------------------------------
# Create splitter
# ---------------------------------------------------

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

# ---------------------------------------------------
# Split the string
# ---------------------------------------------------

chunks = splitter.split_text(text)

# ---------------------------------------------------
# Results
# ---------------------------------------------------

print("=" * 60)
print("Total Chunks")
print("=" * 60)

print(len(chunks))

for i, chunk in enumerate(chunks, start=1):

    print("\n" + "=" * 60)
    print(f"Chunk {i}")
    print("=" * 60)

    print(chunk)