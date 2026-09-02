"""
06_rag_with_sources.py

Learning goal:
    Return both the generated answer
    and the retrieved source documents.

Important:
    The LLM generates the answer.
    The retriever provides the evidence.
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
Answer the question using only the context.

If the answer is not in the context,
say that you don't know.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# Ask question
# ============================================================

question = input("Ask a question: ")


# ============================================================
# Retrieve evidence
# ============================================================

documents = retriever.invoke(question)


# ============================================================
# Convert documents to context
# ============================================================

context = "\n\n".join(
    document.page_content
    for document in documents
)


# ============================================================
# Generate answer
# ============================================================

messages = prompt.invoke(
    {
        "context": context,
        "question": question,
    }
)

response = llm.invoke(messages)


# ============================================================
# Answer
# ============================================================

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)


# ============================================================
# Sources
# ============================================================

print("\n" + "=" * 60)
print("SOURCES USED")
print("=" * 60)

for index, document in enumerate(documents, start=1):

    print(f"\nSource {index}")

    print("Metadata:")
    print(document.metadata)

    print("\nContent:")
    print(document.page_content)