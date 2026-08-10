"""
Module 06 - Chat Models
Lesson 03 - System, Human & AI Messages

Goal
----
- Understand SystemMessage
- Understand HumanMessage
- Understand AIMessage
- Build a conversation manually
- Send the conversation to ChatOllama
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Create Conversation
# -------------------------------------------------

messages = [

    SystemMessage(
        content=(
            "You are a helpful AI teacher. "
            "Explain technical concepts simply."
        )
    ),

    HumanMessage(
        content="What is an embedding?"
    ),

    AIMessage(
        content=(
            "An embedding is a numerical representation "
            "of text or other data."
        )
    ),

    HumanMessage(
        content="Why do we use embeddings in RAG?"
    ),
]


# -------------------------------------------------
# Display Conversation
# -------------------------------------------------

print("=" * 60)
print("CONVERSATION")
print("=" * 60)

for message in messages:

    print(f"\n{message.type.upper()}:")
    print(message.content)


# -------------------------------------------------
# Send Conversation to Model
# -------------------------------------------------

response = llm.invoke(messages)


# -------------------------------------------------
# Display New AI Response
# -------------------------------------------------

print("\n" + "=" * 60)
print("NEW AI RESPONSE")
print("=" * 60)

print(response.content)