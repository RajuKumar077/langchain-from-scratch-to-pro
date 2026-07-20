"""
Lesson 05 - Cosine Similarity

Goal
----
- Learn cosine similarity
- Compare similar and unrelated sentences
"""

from langchain_ollama import OllamaEmbeddings
from math import sqrt

print("=" * 60)
print("Loading Embedding Model")
print("=" * 60)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("✅ Model Loaded")

# -------------------------------------------------
# Sentences
# -------------------------------------------------

sentence1 = "Artificial Intelligence is changing the world."

sentence2 = "AI is transforming our world."

sentence3 = "I love eating pizza."

# -------------------------------------------------
# Generate embeddings
# -------------------------------------------------

vector1 = embeddings.embed_query(sentence1)
vector2 = embeddings.embed_query(sentence2)
vector3 = embeddings.embed_query(sentence3)

print("✅ Embeddings Generated")


# -------------------------------------------------
# Cosine Similarity Function
# -------------------------------------------------

def cosine_similarity(v1, v2):

    # Dot Product
    dot_product = sum(a * b for a, b in zip(v1, v2))

    # Magnitude of first vector
    magnitude1 = sqrt(sum(a * a for a in v1))

    # Magnitude of second vector
    magnitude2 = sqrt(sum(b * b for b in v2))

    return dot_product / (magnitude1 * magnitude2)


# -------------------------------------------------
# Compare
# -------------------------------------------------

similarity12 = cosine_similarity(vector1, vector2)

similarity13 = cosine_similarity(vector1, vector3)

# -------------------------------------------------
# Results
# -------------------------------------------------

print("\n" + "=" * 60)
print("Sentence 1")
print("=" * 60)

print(sentence1)

print("\nSentence 2")
print(sentence2)

print("\nSentence 3")
print(sentence3)

print("\n" + "=" * 60)
print("Cosine Similarity (1 vs 2)")
print("=" * 60)

print(similarity12)

print("\n" + "=" * 60)
print("Cosine Similarity (1 vs 3)")
print("=" * 60)

print(similarity13)