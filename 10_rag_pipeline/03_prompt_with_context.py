"""
03_prompt_with_context.py

Goal:
    Learn how a RAG prompt combines:

        1. User question
        2. Retrieved context
        3. Instructions

This is the "Augmented" part of
Retrieval-Augmented Generation.
"""

from langchain_core.prompts import ChatPromptTemplate


# ============================================================
# Example retrieved context
#
# Normally this comes from PGVector.
# For this exercise, we manually provide it.
# ============================================================

context = """
LangChain is a framework for developing applications
powered by language models.

LangChain provides components for prompts, models,
retrievers, document processing, and chains.
"""


# ============================================================
# User question
# ============================================================

question = "What is LangChain?"


# ============================================================
# Create the RAG prompt
# ============================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful AI assistant.

Answer the user's question using the provided context.

Rules:
- Use the context as your primary source.
- Do not invent information.
- If the context does not contain the answer,
  say that you don't know.

Context:
{context}

Question:
{question}

Answer:
"""
)


# ============================================================
# Create the final prompt
# ============================================================

messages = prompt.invoke(
    {
        "context": context,
        "question": question,
    }
)


# ============================================================
# Display the prompt
# ============================================================

print("=" * 60)
print("PROMPT SENT TO THE LLM")
print("=" * 60)

print(messages)