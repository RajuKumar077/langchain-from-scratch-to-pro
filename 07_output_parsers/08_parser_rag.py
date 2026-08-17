"""
Module 07 - Output Parsers
Lesson 08 - Output Parser with RAG

Goal
----
- Retrieve relevant documents
- Pass retrieved context to a prompt
- Generate an answer with the LLM
- Parse the final answer into a string
"""

from pathlib import Path

from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_postgres import PGVector


# -------------------------------------------------
# PostgreSQL Connection
# -------------------------------------------------

CONNECTION = (
    "postgresql+psycopg://langchain:langchain@localhost:6024/langchain"
)

COLLECTION_NAME = "lesson08"


# -------------------------------------------------
# Embedding Model
# -------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# -------------------------------------------------
# Connect to Vector Store
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
    temperature=0,
)


# -------------------------------------------------
# Create Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Answer the question using ONLY the
        provided context.

        If the answer is not present in the
        context, say:

        "I don't know based on the provided context."
        """
    ),
    (
        "human",
        """
        Context:
        {context}

        Question:
        {question}
        """
    ),
])


# -------------------------------------------------
# Create Output Parser
# -------------------------------------------------

parser = StrOutputParser()


# -------------------------------------------------
# Ask Question
# -------------------------------------------------

question = input("\nAsk a Question: ")


# -------------------------------------------------
# Retrieve Documents
# -------------------------------------------------

documents = retriever.invoke(question)


# -------------------------------------------------
# Create Context
# -------------------------------------------------

context = "\n\n".join(
    document.page_content
    for document in documents
)


# -------------------------------------------------
# Create Prompt Messages
# -------------------------------------------------

messages = prompt.invoke({
    "context": context,
    "question": question,
})


# -------------------------------------------------
# Call LLM
# -------------------------------------------------

try:

    response = llm.invoke(messages)

    # ---------------------------------------------
    # Parse Response
    # ---------------------------------------------

    answer = parser.invoke(response)


    # ---------------------------------------------
    # Display Answer
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("RAG ANSWER")
    print("=" * 60)

    print(answer)


    # ---------------------------------------------
    # Display Retrieved Documents
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("RETRIEVED DOCUMENTS")
    print("=" * 60)

    for i, document in enumerate(documents, start=1):

        print(f"\n--- Document {i} ---")
        print(document.page_content)


except Exception as e:

    print("\n" + "=" * 60)
    print("ERROR")
    print("=" * 60)

    print(e)