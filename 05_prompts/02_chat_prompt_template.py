"""
Module 05 - Prompts
Lesson 02 - ChatPromptTemplate

Goal:
- Understand ChatPromptTemplate
- Understand system and human messages
- Pass variables into chat prompts
"""

from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------
# Create Chat Prompt Template
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant."
    ),
    (
        "human",
        "Explain {topic} in simple terms."
    )
])


# -------------------------------------------------
# Provide a value for the variable
# -------------------------------------------------

messages = prompt.invoke({
    "topic": "Machine Learning"
})


# -------------------------------------------------
# Print the result
# -------------------------------------------------

print("=" * 60)
print("Chat Prompt Template")
print("=" * 60)

print(messages)


# -------------------------------------------------
# Print each message separately
# -------------------------------------------------

print("\n" + "=" * 60)
print("Individual Messages")
print("=" * 60)

for message in messages.messages:
    print("\nType:", message.type)
    print("Content:", message.content)