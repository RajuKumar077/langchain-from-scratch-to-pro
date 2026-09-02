"""
09_rag_conversation.py

Learning goal:
    Combine:

        Conversation history
              +
        RAG context
              +
        LLM

This builds on the memory concepts
you already studied.
"""

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate


POSTGRES_URL = (
    "postgresql+psycopg://langchain:langchain"
    "@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"


# ============================================================
# Models
# ============================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
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


retriever = vector_store.as_retriever(
    search_kwargs={"k": 3}
)


# ============================================================
# Conversation history
# ============================================================

conversation_history = []


# ============================================================
# Prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
Answer using the conversation history and
the retrieved context.

Do not invent information.

If the answer is not available in the context,
say that you don't know.

Conversation history:
{history}

Retrieved context:
{context}

Current question:
{question}

Answer:
"""
)


# ============================================================
# Conversation loop
# ============================================================

while True:

    question = input("\nYou: ")

    if question.lower() in {"exit", "quit"}:
        break


    # ========================================================
    # Retrieve relevant documents
    # ========================================================

    documents = retriever.invoke(question)


    # ========================================================
    # Convert documents into context
    # ========================================================

    context = "\n\n".join(
        document.page_content
        for document in documents
    )


    # ========================================================
    # Convert conversation history to text
    # ========================================================

    history = "\n".join(
        f"User: {user}\nAssistant: {assistant}"
        for user, assistant in conversation_history
    )


    # ========================================================
    # Create prompt
    # ========================================================

    messages = prompt.invoke(
        {
            "history": history,
            "context": context,
            "question": question,
        }
    )


    # ========================================================
    # Generate response
    # ========================================================

    response = llm.invoke(messages)

    answer = response.content


    # ========================================================
    # Display answer
    # ========================================================

    print(f"\nAssistant: {answer}")


    # ========================================================
    # Save conversation
    # ========================================================

    conversation_history.append(
        (question, answer)
    )