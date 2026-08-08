"""
Module 05 - Prompts
Lesson 09 - RAG Prompt

Goal
----
- Create a dedicated RAG prompt
- Provide retrieved context to the LLM
- Instruct the LLM to answer only from context
- Handle questions that cannot be answered
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
# RAG Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful question-answering assistant.

Answer the question using ONLY the information
provided in the context.

Do not use outside knowledge.

If the context does not contain enough information
to answer the question, say:

"I don't know based on the provided context."

Keep your answer clear and concise.

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
# Retrieve Relevant Documents
# -------------------------------------------------

documents = retriever.invoke(question)


# -------------------------------------------------
# Build Context
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
# Display Context
# -------------------------------------------------

print("\n" + "=" * 60)
print("RETRIEVED CONTEXT")
print("=" * 60)

print(context)


# -------------------------------------------------
# Display Final Prompt
# -------------------------------------------------

print("\n" + "=" * 60)
print("FINAL PROMPT")
print("=" * 60)

for message in messages.messages:

    print(f"\n{message.type.upper()}:")
    print(message.content)


# -------------------------------------------------
# Send to LLM
# -------------------------------------------------

response = llm.invoke(messages)


# -------------------------------------------------
# Display Answer
# -------------------------------------------------

print("\n" + "=" * 60)
print("RAG ANSWER")
print("=" * 60)

print(response.content)