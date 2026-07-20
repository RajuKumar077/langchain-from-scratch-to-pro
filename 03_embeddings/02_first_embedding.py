"""
Lesson 02 - Generate Multiple Embeddings

Goal
----
- Generate embeddings for multiple sentences
- Understand that every sentence gets its own vector
"""

from langchain_ollama import OllamaEmbeddings

print("=" * 60)
print("Loading Ollama Embedding Model")
print("=" * 60)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("✅ Model Loaded")

# ----------------------------------------------------
# Sentences
# ----------------------------------------------------

texts = [
    "Artificial Intelligence is transforming healthcare.",
    "Machine Learning is a subset of Artificial Intelligence.",
    "Cats love drinking milk.",
    "Python is a popular programming language."
]

print("\nGenerating embeddings...\n")

# embed_documents() accepts a list of strings
vectors = embeddings.embed_documents(texts)

print("✅ Embeddings Generated")

# ----------------------------------------------------
# Total embeddings
# ----------------------------------------------------

print("\n" + "=" * 60)
print("Total Embeddings")
print("=" * 60)

print(len(vectors))

# ----------------------------------------------------
# Display information
# ----------------------------------------------------

for i, vector in enumerate(vectors):

    print("\n" + "=" * 60)
    print(f"Sentence {i+1}")
    print("=" * 60)

    print(texts[i])

    print("\nVector Length:")
    print(len(vector))

    print("\nFirst 10 Numbers:")
    print(vector[:10])