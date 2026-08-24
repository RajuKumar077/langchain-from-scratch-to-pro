"""
Module 09 - Memory
Lesson 08 - Memory with RAG

Goal
----
- Combine conversation memory with RAG
- Retrieve relevant documents
- Preserve conversation history
- Inject retrieved context into the prompt
- Send everything to the LLM
"""

from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter,
)

from langchain_postgres import PGVector

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)


# =================================================
# PostgreSQL
# =================================================

CONNECTION = (
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)

COLLECTION_NAME = "memory_rag"


# =================================================
# Locate Data File
# =================================================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_FILE = (
    BASE_DIR / "data" / "sample.txt"
)


# =================================================
# Load Documents
# =================================================

loader = TextLoader(
    str(DATA_FILE)
)

documents = loader.load()


print("=" * 60)
print("DOCUMENTS LOADED")
print("=" * 60)

print(
    "Documents:",
    len(documents)
)


# =================================================
# Split Documents
# =================================================

splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=50,
)

chunks = splitter.split_documents(
    documents
)


print(
    "Chunks:",
    len(chunks)
)


# =================================================
# Embedding Model
# =================================================

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# =================================================
# Create Vector Store
# =================================================

vector_store = PGVector.from_documents(
    documents=chunks,
    embedding=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)


print(
    "Vector store ready."
)


# =================================================
# Create Retriever
# =================================================

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# =================================================
# Create Chat Model
# =================================================

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# =================================================
# Prompt
# =================================================

prompt = ChatPromptTemplate.from_messages([

    (
        "system",
        """
        You are a helpful RAG assistant.

        Answer the user's question using the
        retrieved context.

        Conversation history can be used to
        understand references from previous turns.

        Do not invent information.

        Retrieved Context:

        {context}
        """
    ),

    MessagesPlaceholder(
        variable_name="history"
    ),

    (
        "human",
        "{question}"
    ),
])


# =================================================
# Session Store
# =================================================

store = {}


# =================================================
# Get Session History
# =================================================

def get_session_history(
    session_id: str,
) -> BaseChatMessageHistory:

    if session_id not in store:

        store[session_id] = (
            InMemoryChatMessageHistory()
        )

    return store[session_id]


# =================================================
# RAG + Memory Function
# =================================================

def rag_chat(
    question: str,
    session_id: str,
):

    # ---------------------------------------------
    # Retrieve Documents
    # ---------------------------------------------

    retrieved_documents = retriever.invoke(
        question
    )


    # ---------------------------------------------
    # Combine Retrieved Documents
    # ---------------------------------------------

    context = "\n\n".join(
        document.page_content
        for document in retrieved_documents
    )


    # ---------------------------------------------
    # Get Conversation History
    # ---------------------------------------------

    history = get_session_history(
        session_id
    )


    # ---------------------------------------------
    # Build Prompt
    # ---------------------------------------------

    messages = prompt.invoke({
        "context": context,
        "history": history.messages,
        "question": question,
    })


    # ---------------------------------------------
    # Call LLM
    # ---------------------------------------------

    response = llm.invoke(
        messages
    )


    # ---------------------------------------------
    # Store Conversation
    # ---------------------------------------------

    history.add_user_message(
        question
    )

    history.add_ai_message(
        response.content
    )


    return response.content, retrieved_documents


# =================================================
# Start Conversation
# =================================================

print("\n" + "=" * 60)
print("MEMORY + RAG")
print("=" * 60)


session_id = "user_001"


# =================================================
# First Question
# =================================================

question = input(
    "\nAsk a Question: "
)


answer, documents = rag_chat(
    question,
    session_id,
)


print("\n" + "=" * 60)
print("ANSWER")
print("=" * 60)

print(answer)


# =================================================
# Second Question
# =================================================

question = input(
    "\nAsk a Follow-up Question: "
)


answer, documents = rag_chat(
    question,
    session_id,
)


print("\n" + "=" * 60)
print("FOLLOW-UP ANSWER")
print("=" * 60)

print(answer)


# =================================================
# Show Retrieved Documents
# =================================================

print("\n" + "=" * 60)
print("RETRIEVED DOCUMENTS")
print("=" * 60)


for index, document in enumerate(
    documents,
    start=1,
):

    print(
        f"\n--- Document {index} ---"
    )

    print(
        document.page_content
    )


# =================================================
# Show Conversation History
# =================================================

print("\n" + "=" * 60)
print("CONVERSATION HISTORY")
print("=" * 60)


history = get_session_history(
    session_id
)


for message in history.messages:

    print(
        f"\n{message.type.upper()}:"
    )

    print(
        message.content
    )