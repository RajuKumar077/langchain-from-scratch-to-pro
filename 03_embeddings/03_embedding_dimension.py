"""
Lesson 03 - Understanding Embedding Dimensions

Goal
----
- Understand what an embedding dimension is
- Verify that every text returns the same vector length
- See that text length does NOT affect embedding size
"""

from langchain_ollama import OllamaEmbeddings

print("=" * 60)
print("Loading Embedding Model")
print("=" * 60)

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

print("✅ Model Loaded")

# -------------------------------------------------
# Different lengths of text
# -------------------------------------------------

texts = [
    "AI",
    "Artificial Intelligence",
    "Artificial Intelligence is changing the world.",
    """
Artificial Intelligence is transforming industries like
healthcare, finance, education, manufacturing, and transportation.
Large Language Models are becoming increasingly popular because
they understand and generate human language.
"""
]

# -------------------------------------------------
# Generate embedding for each text
# -------------------------------------------------

for i, text in enumerate(texts):

    print("\n" + "=" * 60)
    print(f"Example {i+1}")
    print("=" * 60)

    print("\nText:")
    print(text)

    vector = embeddings.embed_query(text)

    print("\nText Length (Characters):")
    print(len(text))

    print("\nEmbedding Length:")
    print(len(vector))

    print("\nFirst 5 Numbers:")
    print(vector[:5])