"""
Module 05 - Prompts
Lesson 04 - System & Human Messages

Goal:
- Understand SystemMessage
- Understand HumanMessage
- Understand their different roles
"""

from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------
# Create Chat Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert Python teacher. "
        "Explain technical concepts clearly and simply."
    ),
    (
        "human",
        "Explain {topic}."
    )
])


# -------------------------------------------------
# Provide User Input
# -------------------------------------------------

messages = prompt.invoke({
    "topic": "Embeddings"
})


# -------------------------------------------------
# Print Messages
# -------------------------------------------------

print("=" * 60)
print("System & Human Messages")
print("=" * 60)

for message in messages.messages:

    print("\nMessage Type:")
    print(message.type)

    print("\nMessage Content:")
    print(message.content)

    print("-" * 60)