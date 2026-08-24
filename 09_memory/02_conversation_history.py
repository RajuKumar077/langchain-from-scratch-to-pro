"""
Module 09 - Memory
Lesson 02 - Conversation History

Goal
----
- Understand conversation history
- Store Human and AI messages
- Reuse the complete history
- Build a simple multi-turn conversation
"""

from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# -------------------------------------------------
# Conversation History
# -------------------------------------------------

history = []


# -------------------------------------------------
# Function to Ask the Model
# -------------------------------------------------

def chat(user_input):

    # Add user's message
    history.append(
        HumanMessage(content=user_input)
    )

    # Send complete conversation
    response = llm.invoke(history)

    # Save AI response
    history.append(response)

    return response.content


# -------------------------------------------------
# Conversation
# -------------------------------------------------

print("=" * 60)
print("CONVERSATION")
print("=" * 60)


answer = chat("My name is Raju.")

print("\nAI:", answer)


answer = chat("I am learning LangChain.")

print("\nAI:", answer)


answer = chat("What am I learning?")

print("\nAI:", answer)


answer = chat("What is my name?")

print("\nAI:", answer)


# -------------------------------------------------
# Display History
# -------------------------------------------------

print("\n" + "=" * 60)
print("FULL CONVERSATION HISTORY")
print("=" * 60)

for message in history:

    print(
        f"\n{message.type.upper()}:"
    )

    print(message.content)