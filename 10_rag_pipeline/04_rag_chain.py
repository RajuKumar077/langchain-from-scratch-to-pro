"""
04_rag_chain.py

Goal:
    Build our first complete RAG chain manually.

Flow:

    Question
       ↓
    Retriever
       ↓
    Documents
       ↓
    Context
       ↓
    Prompt
       ↓
    LLM
       ↓
    Answer
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
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
# Vector Store
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
Answer the question using the provided context.

If the answer is not present in the context,
say "I don't know based on the provided context."

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# User Question
# ============================================================

question = input("Ask a question: ")


# ============================================================
# STEP 1: Retrieve
# ============================================================

documents = retriever.invoke(question)


# ============================================================
# STEP 2: Convert Documents into Context
# ============================================================

context = "\n\n".join(
    document.page_content
    for document in documents
)


# ============================================================
# STEP 3: Create Prompt
# ============================================================

messages = prompt.invoke(
    {
        "context": context,
        "question": question,
    }
)


# ============================================================
# STEP 4: Generate Answer
# ============================================================

response = llm.invoke(messages)


# ============================================================
# STEP 5: Display Result
# ============================================================

print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(response.content)