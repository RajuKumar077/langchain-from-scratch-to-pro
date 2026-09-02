"""
05_rag_chain_lcel.py

Learning goal:
    Build RAG using LCEL.

LCEL = LangChain Expression Language

The | operator connects components together.

Pipeline:

    Question
       ↓
    Retriever
       ↓
    format_docs
       ↓
    Prompt
       ↓
    Mistral
       ↓
    String output
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


# ============================================================
# Configuration
# ============================================================

POSTGRES_URL = (
    "postgresql+psycopg://langchain:langchain"
    "@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"


# ============================================================
# Embeddings
# ============================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# ============================================================
# PGVector
# ============================================================

vector_store = PGVector(
    embeddings=embeddings,
    collection_name=COLLECTION_NAME,
    connection=POSTGRES_URL,
    use_jsonb=True,
)


# ============================================================
# Retriever
# ============================================================

retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# LLM
# ============================================================

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# ============================================================
# Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
Answer the question using ONLY the provided context.

If the answer is not available in the context,
say that you don't know.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# Format retrieved Documents
# ============================================================

def format_docs(documents):
    """
    Convert LangChain Document objects
    into one text string.
    """

    return "\n\n".join(
        document.page_content
        for document in documents
    )


# ============================================================
# Build the RAG chain
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
# Run
# ============================================================

question = input("Ask a question: ")

answer = rag_chain.invoke(question)


print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(answer)