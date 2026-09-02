"""
01_basic_rag.py

Goal:
    Understand the basic RAG flow.

RAG = Retrieval-Augmented Generation

Flow:

    Question
        ↓
    Retriever
        ↓
    Relevant Documents
        ↓
    Context
        ↓
    Prompt
        ↓
    LLM
        ↓
    Answer

This example intentionally keeps the pipeline simple.
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# 1. Configuration
# ============================================================

POSTGRES_URL = (
    "postgresql+psycopg://langchain:langchain"
    "@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "mistral:latest"


# ============================================================
# 2. Create the embedding model
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# 3. Connect to the existing PGVector collection
# ============================================================

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=POSTGRES_URL,
    use_jsonb=True,
)


# ============================================================
# 4. Create a retriever
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# 5. Create the local LLM
# ============================================================

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
)


# ============================================================
# 6. Create the RAG prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Answer the question using ONLY the context provided below.

If the answer cannot be found in the context,
say that you don't know.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# 7. Ask a question
# ============================================================

question = input("Ask a question: ")


# ============================================================
# 8. Retrieve relevant documents
# ============================================================

documents = retriever.invoke(question)


# ============================================================
# 9. Convert Documents into plain text
# ============================================================

context = "\n\n".join(
    document.page_content
    for document in documents
)


# ============================================================
# 10. Build the final prompt
# ============================================================

messages = prompt.invoke(
    {
        "context": context,
        "question": question,
    }
)


# ============================================================
# 11. Generate the answer
# ============================================================

response = llm.invoke(messages)


# ============================================================
# 12. Display the answer
# ============================================================

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)