"""
Lesson 01 - Why Split Documents?

Goal:
- Understand why we split documents.
- Understand the limitation of using one huge document.
"""

from langchain_core.documents import Document

# ----------------------------------------------------
# Create one BIG document
# ----------------------------------------------------

document = Document(
    page_content="""
Artificial Intelligence (AI) is transforming industries across the world.

Machine Learning is a subset of Artificial Intelligence.
It allows computers to learn patterns from data.

Deep Learning is a subset of Machine Learning.
It uses neural networks with many hidden layers.

Computer Vision helps computers understand images.

Natural Language Processing (NLP) helps computers understand text.

Large Language Models like GPT, Llama and Mistral
are trained on huge amounts of text.

Vector Databases store embeddings for semantic search.

LangChain helps connect LLMs with external knowledge.

RAG stands for Retrieval Augmented Generation.
It allows an LLM to answer using your own data.

Embeddings convert text into numbers.

Similarity Search finds the most relevant information.

Chunking divides large documents into smaller pieces.
""",

    metadata={
        "source": "AI Notes",
        "chapter": 1
    }
)

# ----------------------------------------------------
# Print information
# ----------------------------------------------------

print("=" * 60)
print("Document")
print("=" * 60)

print(document)

print()

print("=" * 60)
print("Total Characters")
print("=" * 60)

print(len(document.page_content))