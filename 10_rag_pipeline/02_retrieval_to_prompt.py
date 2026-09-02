"""
02_retrieval_to_prompt.py

Goal:
    Understand how retrieved LangChain Documents
    are converted into context for the LLM.

Important concept:

    Retriever returns Document objects.

    The LLM does NOT directly receive those Document objects.

    We extract:
        document.page_content

    and combine the text into a context string.
"""

from langchain_ollama import OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# Configuration
# ============================================================

POSTGRES_URL = (
    "postgresql+psycopg://langchain:langchain"
    "@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ============================================================
# Connect to PGVector
# ============================================================

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=POSTGRES_URL,
    use_jsonb=True,
)


# ============================================================
# Create Retriever
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# Ask a question
# ============================================================

question = input("Ask a question: ")


# ============================================================
# Retrieve documents
# ============================================================

documents = retriever.invoke(question)


# ============================================================
# Inspect what the retriever returned
# ============================================================

print("\n" + "=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)

for index, document in enumerate(documents, start=1):

    print(f"\n--- Document {index} ---")

    print("Content:")
    print(document.page_content)

    print("\nMetadata:")
    print(document.metadata)


# ============================================================
# Convert documents into context
# ============================================================

context = "\n\n".join(
    document.page_content
    for document in documents
)


# ============================================================
# Create prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
Use the following context to answer the question.

Context:
{context}

Question:
{question}
"""
)


# ============================================================
# Insert context + question into the prompt
# ============================================================

final_prompt = prompt.invoke(
    {
        "context": context,
        "question": question,
    }
)


# ============================================================
# Display what would be sent to the LLM
# ============================================================

print("\n" + "=" * 60)
print("FINAL PROMPT")
print("=" * 60)

print(final_prompt)