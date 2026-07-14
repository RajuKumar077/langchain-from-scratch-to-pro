"""
Lesson 10 - Best Practices for Text Splitting

Goal:
- Learn the recommended practices for chunking documents.
- Understand how to choose chunk sizes.
- Avoid common mistakes in RAG applications.
"""

print("=" * 70)
print("Text Splitting - Best Practices")
print("=" * 70)

best_practices = [

    # General Rules
    "1. Always split large documents before creating embeddings.",

    "2. Never create one embedding for an entire book or large PDF.",

    "3. Smaller chunks improve retrieval accuracy.",

    "4. Larger chunks preserve more context but may reduce search precision.",

    "5. Use chunk_overlap to avoid losing important context between chunks.",

    # Splitter Choice
    "6. RecursiveCharacterTextSplitter is the default choice for most RAG applications.",

    "7. Customize separators only when your document has a specific structure (Markdown, Code, HTML, etc.).",

    # Metadata
    "8. Prefer split_documents() because it preserves metadata.",

    "9. Use split_text() only when working with a plain Python string.",

    # Chunk Size
    "10. Start with chunk_size between 300 and 1000 characters for general documents.",

    "11. Start with chunk_overlap between 50 and 200 characters.",

    "12. Experiment with different chunk sizes depending on your use case.",

    # RAG Pipeline
    "13. Chunking always happens BEFORE embeddings are created.",

    "14. Every chunk usually becomes one embedding vector.",

    "15. One 300-page PDF usually produces hundreds or thousands of embeddings.",

    # Performance
    "16. Smaller chunks are faster and cheaper to embed.",

    "17. More chunks require more storage in the vector database.",

    "18. Balance chunk size, overlap, cost, and retrieval quality.",

    # Real Projects
    "19. In most LangChain RAG projects you'll use split_documents().",

    "20. Always test your chunking strategy before building a production RAG system."

]

for practice in best_practices:
    print(practice)

print("\n" + "=" * 70)
print("Quick Revision")
print("=" * 70)

print("""
✔ Loader → Loads the document
✔ Document → Stores page_content + metadata
✔ Text Splitter → Breaks large documents into smaller chunks
✔ Each Chunk → Still a Document (when using split_documents())
✔ Embedding Model → Converts each chunk into a vector
✔ PGVector → Stores those vectors
✔ Retriever → Finds the most similar vectors
✔ LLM → Uses retrieved chunks to answer the user's question
""")