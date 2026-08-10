"""
Module 06 - Chat Models
Lesson 08 - Chat Model + Prompt

Goal
----
- Create a ChatPromptTemplate
- Create a ChatOllama model
- Connect Prompt + Chat Model
- Invoke the complete pipeline
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# -------------------------------------------------
# Create Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI teacher. "
        "Explain technical concepts simply."
    ),
    (
        "human",
        "Explain {topic} in simple terms."
    )
])


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Provide Input
# -------------------------------------------------

topic = input("\nEnter a topic: ")


# -------------------------------------------------
# Create Messages from Prompt
# -------------------------------------------------

messages = prompt.invoke({
    "topic": topic
})


# -------------------------------------------------
# Display Messages
# -------------------------------------------------

print("\n" + "=" * 60)
print("PROMPT MESSAGES")
print("=" * 60)

for message in messages.messages:

    print(f"\n{message.type.upper()}:")
    print(message.content)


# -------------------------------------------------
# Send Messages to Chat Model
# -------------------------------------------------

response = llm.invoke(messages)


# -------------------------------------------------
# Display AI Response
# -------------------------------------------------

print("\n" + "=" * 60)
print("AI RESPONSE")
print("=" * 60)

print(response.content)