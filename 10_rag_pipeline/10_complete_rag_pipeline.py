"""
10_complete_rag_pipeline.py

FINAL PROJECT
=============

A complete local RAG application.

Technology:

    LLM:
        Ollama
        qwen3:8b

    Embeddings:
        Ollama
        nomic-embed-text

    Vector Store:
        PostgreSQL + PGVector

    Collection:
        complete_pipeline

    Framework:
        LangChain

Pipeline:

    User Question
          ↓
    Query Embedding
          ↓
    PGVector Retrieval
          ↓
    Relevant Documents
          ↓
    Context Formattingahat 
          ↓
    Prompt
          ↓
    Qwen3
          ↓
    Answer + Sources
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# CONFIGURATION
# ============================================================

POSTGRES_URL = (
    "postgresql+psycopg://langchain:langchain"
    "@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"

EMBEDDING_MODEL = "nomic-embed-text"

LLM_MODEL = "qwen3:8b"


# ============================================================
# EMBEDDING MODEL
# ============================================================

embeddings = OllamaEmbeddings(
    model=EMBEDDING_MODEL
)


# ============================================================
# VECTOR STORE
# ============================================================

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=POSTGRES_URL,
    use_jsonb=True,
)


# ============================================================
# RETRIEVER
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 4}
)


# ============================================================
# LOCAL LLM
# ============================================================

llm = ChatOllama(
    model=LLM_MODEL,
    temperature=0,
)


# ============================================================
# PROMPT
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a reliable document question-answering assistant.

Answer the user's question using ONLY the provided context.

Rules:

1. Do not invent facts.
2. Do not use outside knowledge.
3. If the context does not contain the answer,
   say:

   "I don't know based on the provided context."

4. Keep the answer clear and concise.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# DOCUMENT FORMATTER
# ============================================================

def format_docs(documents):
    """
    Convert LangChain Document objects into
    one text block that can be inserted into
    the prompt.
    """

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ============================================================
# RAG CHAIN
# ============================================================

rag_chain = (
    {
        "context": retriever | format_docs,
        "question": RunnablePassthrough(),
    }
    | prompt
    | llm
    | StrOutputParser()
)


# ============================================================
# USER QUESTION
# ============================================================

question = input("Ask a question: ")


# ============================================================
# GENERATE ANSWER
# ============================================================

answer = rag_chain.invoke(question)


# ============================================================
# DISPLAY ANSWER
# ============================================================

print("\n" + "=" * 70)
print("RAG ANSWER")
print("=" * 70)

print(answer)


# ============================================================
# RETRIEVE SOURCES SEPARATELY
#
# The RAG chain above only returns the final text.
# We retrieve the documents again so we can show
# the evidence used by the system.
# ============================================================

documents = retriever.invoke(question)


print("\n" + "=" * 70)
print("RETRIEVED SOURCES")
print("=" * 70)


for index, document in enumerate(documents, start=1):

    print(f"\n--- Source {index} ---")

    print("Metadata:")
    print(document.metadata)

    print("\nContent:")
    print(document.page_content[:500])


print("\n" + "=" * 70)
print("RAG PIPELINE COMPLETE")
print("=" * 70)