"""
08_rag_no_context.py

Learning goal:
    Understand that retrieval does NOT guarantee
    that the answer exists in our documents.

We explicitly instruct the LLM to avoid hallucinating.
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate


POSTGRES_URL = (
    "postgresql+psycopg://langchain:langchain"
    "@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"


embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=POSTGRES_URL,
    use_jsonb=True,
)


retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


prompt = ChatPromptTemplate.from_template(
    """
You are a grounded RAG assistant.

Use ONLY the provided context.

If the context does not contain enough
information to answer the question, say:

"I don't know based on the provided context."

Do not use your own general knowledge.

Context:
{context}

Question:
{question}

Answer:
"""
)


question = input("Ask a question: ")


# ============================================================
# Retrieve
# ============================================================

documents = retriever.invoke(question)


# ============================================================
# Show what was retrieved
# ============================================================

print("\n" + "=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)

for document in documents:

    print("\n" + document.page_content)


# ============================================================
# Build context
# ============================================================

context = "\n\n".join(
    document.page_content
    for document in documents
)


# ============================================================
# Generate grounded answer
# ============================================================

messages = prompt.invoke(
    {
        "context": context,
        "question": question,
    }
)

response = llm.invoke(messages)


print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)