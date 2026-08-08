"""
Module 05 - Prompts
Lesson 07 - Retrieval + Prompt

Goal:
- Connect PGVector Retriever with a Prompt
- Retrieve relevant documents
- Put retrieved documents into the prompt
- Send the prompt to the LLM
"""

from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_postgres import PGVector
from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------
# PostgreSQL Connection
# -------------------------------------------------

CONNECTION = (
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)

COLLECTION_NAME = "complete_pipeline"


# -------------------------------------------------
# Embedding Model
# -------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# -------------------------------------------------
# Connect to PGVector
# -------------------------------------------------

vector_store = PGVector(
    embeddings=embeddings,
    connection=CONNECTION,
    collection_name=COLLECTION_NAME,
)


# -------------------------------------------------
# Create Retriever
# -------------------------------------------------

retriever = vector_store.as_retriever(
    search_kwargs={
        "k": 3
    }
)


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Create RAG Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.

Answer the user's question using ONLY the provided context.

If the answer cannot be found in the context,
say that you don't know.

Context:
{context}
"""
    ),
    (
        "human",
        "{question}"
    )
])


# -------------------------------------------------
# Ask Question
# -------------------------------------------------

question = input("\nAsk a Question: ")


# -------------------------------------------------
# Retrieve Documents
# -------------------------------------------------

documents = retriever.invoke(question)


# -------------------------------------------------
# Convert Documents into Context
# -------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in documents
)


# -------------------------------------------------
# Create Final Prompt
# -------------------------------------------------

messages = prompt.invoke({
    "context": context,
    "question": question
})


# -------------------------------------------------
# Display Retrieved Context
# -------------------------------------------------

print("\n" + "=" * 60)
print("Retrieved Context")
print("=" * 60)

print(context)


# -------------------------------------------------
# Send Prompt to LLM
# -------------------------------------------------

response = llm.invoke(messages)


# -------------------------------------------------
# Display Answer
# -------------------------------------------------

print("\n" + "=" * 60)
print("AI Answer")
print("=" * 60)

print(response.content)