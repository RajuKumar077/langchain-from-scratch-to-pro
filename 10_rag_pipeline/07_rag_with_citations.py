"""
07_rag_with_citations.py

Learning goal:
    Add source labels to retrieved context.

Example:

    [Source 1]
    Machine Learning is...

    [Source 2]
    Deep Learning is...

The LLM can reference these labels.
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
Answer the question using the provided sources.

Use source labels when possible.

If the answer cannot be found in the sources,
say that you don't know.

Sources:
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
# Add source labels
# ============================================================

source_blocks = []

for index, document in enumerate(documents, start=1):

    source_blocks.append(
        f"[Source {index}]\n"
        f"{document.page_content}"
    )


context = "\n\n".join(source_blocks)


# ============================================================
# Generate
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


print("\n" + "=" * 60)
print("SOURCE METADATA")
print("=" * 60)

for index, document in enumerate(documents, start=1):

    print(f"[Source {index}]")
    print(document.metadata)