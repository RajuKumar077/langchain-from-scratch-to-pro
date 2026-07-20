"""
Lesson 01 - What is an Embedding?

Goal
----
- Generate your first embedding
- Understand what an embedding looks like
- See that text becomes numbers
"""

# Import Ollama Embeddings
from langchain_ollama import OllamaEmbeddings

print("=" * 60)
print("Creating Embedding Model")
print("=" * 60)

# Load the local embedding model
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("✅ Embedding model loaded!")

# -------------------------------------------------
# Sentence to convert into an embedding
# -------------------------------------------------

text = "Artificial Intelligence is changing the world."

print("\nOriginal Text:")
print(text)

# -------------------------------------------------
# Generate embedding
# -------------------------------------------------

print("\nGenerating embedding...\n")

vector = embeddings.embed_query(text)

print("✅ Embedding Generated!")

# -------------------------------------------------
# Information about the embedding
# -------------------------------------------------

print("\n" + "=" * 60)
print("Embedding Type")
print("=" * 60)

print(type(vector))

print("\n" + "=" * 60)
print("Embedding Length")
print("=" * 60)

print(len(vector))

print("\n" + "=" * 60)
print("First 20 Numbers")
print("=" * 60)

print(vector[:20])