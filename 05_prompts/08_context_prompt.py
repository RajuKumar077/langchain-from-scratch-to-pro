"""
Module 05 - Prompts
Lesson 08 - Context Prompt

Goal
----
- Retrieve relevant documents
- Extract their page_content
- Build a context string
- Insert the context into a prompt
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
# Create LLM
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Create Context Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant.

Use the following context to answer the question.

Context:
{context}

If the answer is not present in the context,
say that you don't know.
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
# Build Context
# -------------------------------------------------

context_parts = []

for document in documents:
    context_parts.append(document.page_content)

context = "\n\n".join(context_parts)


# -------------------------------------------------
# Show Context
# -------------------------------------------------

print("\n" + "=" * 60)
print("CONTEXT SENT TO LLM")
print("=" * 60)

print(context)


# -------------------------------------------------
# Create Final Prompt
# -------------------------------------------------

messages = prompt.invoke({
    "context": context,
    "question": question
})


# -------------------------------------------------
# Show Prompt
# -------------------------------------------------

print("\n" + "=" * 60)
print("FINAL PROMPT")
print("=" * 60)

for message in messages.messages:

    print(f"\n{message.type.upper()}:")
    print(message.content)


# -------------------------------------------------
# Call LLM
# -------------------------------------------------

response = llm.invoke(messages)


# -------------------------------------------------
# Display Answer
# -------------------------------------------------

print("\n" + "=" * 60)
print("AI ANSWER")
print("=" * 60)

print(response.content)