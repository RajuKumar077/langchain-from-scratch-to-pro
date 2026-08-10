"""
Module 06 - Chat Models
Lesson 10 - Chat Model Best Practices

Goal
----
- Configure ChatOllama properly
- Use structured messages
- Keep temperature appropriate
- Handle errors
- Keep model interaction clean
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# -------------------------------------------------
# Get User Question
# -------------------------------------------------

question = input("\nAsk a Question: ")


# -------------------------------------------------
# Create Structured Messages
# -------------------------------------------------

messages = [
    SystemMessage(
        content=(
            "You are a helpful AI assistant. "
            "Give clear and concise answers. "
            "If you are unsure, say that you are unsure."
        )
    ),
    HumanMessage(
        content=question
    ),
]


# -------------------------------------------------
# Call Model Safely
# -------------------------------------------------

try:

    response = llm.invoke(messages)

    # ---------------------------------------------
    # Display Response
    # ---------------------------------------------

    print("\n" + "=" * 60)
    print("AI RESPONSE")
    print("=" * 60)

    print(response.content)


except Exception as e:

    print("\n" + "=" * 60)
    print("MODEL ERROR")
    print("=" * 60)

    print("Unable to generate a response.")
    print(f"Details: {e}")