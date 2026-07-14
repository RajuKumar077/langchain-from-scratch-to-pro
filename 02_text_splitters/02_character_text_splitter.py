"""
Lesson 02 - CharacterTextSplitter

Goal:
- Learn how CharacterTextSplitter works.
- Split one long string into multiple chunks.
"""

from langchain_text_splitters import CharacterTextSplitter

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
# Create Text Splitter
# ------------------------------------------------

splitter = CharacterTextSplitter(

    separator="\n",     # Split using new lines

    chunk_size=80,      # Max characters per chunk

    chunk_overlap=0     # No overlap for now
)

# ------------------------------------------------
# Split the text
# ------------------------------------------------

chunks = splitter.split_text(text)

# ------------------------------------------------
# Print Results
# ------------------------------------------------

print("\n")
print("=" * 60)
print("Total Chunks")
print("=" * 60)

print(len(chunks))

# ------------------------------------------------
# Print each chunk
# ------------------------------------------------

for i, chunk in enumerate(chunks, start=1):

    print("\n" + "=" * 60)
    print(f"Chunk {i}")
    print("=" * 60)

    print(chunk)

    print(f"\nCharacters : {len(chunk)}")