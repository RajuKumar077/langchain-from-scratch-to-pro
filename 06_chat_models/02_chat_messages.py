"""
Module 06 - Chat Models
Lesson 02 - Chat Messages

Goal
----
- Understand SystemMessage
- Understand HumanMessage
- Understand AIMessage
- Send structured messages to ChatOllama
"""

# pyrefly: ignore [missing-import]
from langchain_ollama import ChatOllama
# pyrefly: ignore [missing-import]
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
)


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Create Messages
# -------------------------------------------------

messages = [
    SystemMessage(
        content="You are a helpful AI teacher. "
                "Explain concepts in simple language."
    ),

    HumanMessage(
        content="What is a vector database?"
    ),
]


# -------------------------------------------------
# Send Messages to Model
# -------------------------------------------------

response = llm.invoke(messages)


# -------------------------------------------------
# Display Response
# -------------------------------------------------

print("=" * 60)
print("CHAT RESPONSE")
print("=" * 60)

print(response.content)


# -------------------------------------------------
# Inspect AIMessage
# -------------------------------------------------

print("\n" + "=" * 60)
print("RESPONSE TYPE")
print("=" * 60)

print(type(response))

print("\nMessage Type:")
print(response.type)