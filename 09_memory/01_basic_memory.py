"""
Module 09 - Memory
Lesson 01 - Basic Memory

Goal
----
- Understand conversation memory
- Store previous messages
- Send conversation history back to the LLM
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# -------------------------------------------------
# Create Conversation History
# -------------------------------------------------

history = []


# -------------------------------------------------
# First User Message
# -------------------------------------------------

user_message = HumanMessage(
    content="My name is Raju."
)

history.append(user_message)


# -------------------------------------------------
# Send History to LLM
# -------------------------------------------------

response = llm.invoke(history)

history.append(response)


# -------------------------------------------------
# Display Response
# -------------------------------------------------

print("=" * 60)
print("FIRST RESPONSE")
print("=" * 60)

print(response.content)


# -------------------------------------------------
# Second User Message
# -------------------------------------------------

user_message = HumanMessage(
    content="What is my name?"
)

history.append(user_message)


# -------------------------------------------------
# Send Complete History
# -------------------------------------------------

response = llm.invoke(history)

history.append(response)


# -------------------------------------------------
# Display Response
# -------------------------------------------------

print("\n" + "=" * 60)
print("SECOND RESPONSE")
print("=" * 60)

print(response.content)


# -------------------------------------------------
# Display Conversation History
# -------------------------------------------------

print("\n" + "=" * 60)
print("CONVERSATION HISTORY")
print("=" * 60)

for message in history:

    print(
        f"{message.type}: {message.content}"
    )