"""
Module 09 - Memory
Lesson 03 - Message History

Goal
----
- Understand BaseChatMessageHistory
- Store HumanMessage and AIMessage
- Add messages to history
- Read messages from history
"""

from langchain_ollama import ChatOllama

from langchain_core.messages import (
    HumanMessage,
)

from langchain_core.chat_history import InMemoryChatMessageHistory


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# -------------------------------------------------
# Create Message History
# -------------------------------------------------

history = InMemoryChatMessageHistory()


# -------------------------------------------------
# Function to Chat
# -------------------------------------------------

def chat(user_input):

    # Add user message
    history.add_user_message(
        user_input
    )

    # Get all previous messages
    messages = history.messages

    # Send conversation to LLM
    response = llm.invoke(messages)

    # Add AI response
    history.add_ai_message(
        response.content
    )

    return response.content


# -------------------------------------------------
# Conversation
# -------------------------------------------------

print("=" * 60)
print("CONVERSATION")
print("=" * 60)


answer = chat(
    "My name is Raju."
)

print("\nAI:", answer)


answer = chat(
    "I am learning LangChain."
)

print("\nAI:", answer)


answer = chat(
    "What am I learning?"
)

print("\nAI:", answer)


answer = chat(
    "What is my name?"
)

print("\nAI:", answer)


# -------------------------------------------------
# Display Stored Messages
# -------------------------------------------------

print("\n" + "=" * 60)
print("MESSAGE HISTORY")
print("=" * 60)

for message in history.messages:

    print(
        f"\nType: {message.type}"
    )

    print(
        f"Content: {message.content}"
    )