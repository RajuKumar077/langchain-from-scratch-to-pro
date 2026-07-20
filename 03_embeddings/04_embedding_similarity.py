"""
Lesson 04 - Comparing Embeddings

Goal
----
- Generate embeddings for multiple sentences
- Compare embeddings
- Understand that similar sentences produce similar vectors
"""

from langchain_ollama import OllamaEmbeddings
import math

print("=" * 60)
print("Loading Embedding Model")
print("=" * 60)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("✅ Model Loaded")

# ----------------------------------------------------
# Example Sentences
# ----------------------------------------------------

sentence1 = "Artificial Intelligence is changing the world."

sentence2 = "AI is transforming our world."

sentence3 = "I love eating pizza."

# ----------------------------------------------------
# Generate Embeddings
# ----------------------------------------------------

vector1 = embeddings.embed_query(sentence1)
vector2 = embeddings.embed_query(sentence2)
vector3 = embeddings.embed_query(sentence3)

print("\n✅ Embeddings Generated")

# ----------------------------------------------------
# Function to calculate Euclidean Distance
# (Temporary method just for visualization)
# ----------------------------------------------------

def euclidean_distance(v1, v2):
    return math.sqrt(
        sum((a - b) ** 2 for a, b in zip(v1, v2))
    )

# ----------------------------------------------------
# Compare Similar Sentences
# ----------------------------------------------------

distance12 = euclidean_distance(vector1, vector2)

distance13 = euclidean_distance(vector1, vector3)

# ----------------------------------------------------
# Display Results
# ----------------------------------------------------

print("\n" + "=" * 60)
print("Sentence 1")
print("=" * 60)
print(sentence1)

print("\n" + "=" * 60)
print("Sentence 2")
print("=" * 60)
print(sentence2)

print("\n" + "=" * 60)
print("Sentence 3")
print("=" * 60)
print(sentence3)

print("\n" + "=" * 60)
print("Distance Between Sentence 1 & Sentence 2")
print("=" * 60)

print(distance12)

print("\n" + "=" * 60)
print("Distance Between Sentence 1 & Sentence 3")
print("=" * 60)

print(distance13)