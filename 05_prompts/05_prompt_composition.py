"""
Module 05 - Prompts
Lesson 05 - Prompt Composition

Goal:
- Build prompts from multiple parts
- Reuse prompt components
- Understand the | operator
"""

from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------
# Create individual prompt parts
# -------------------------------------------------

system_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert AI/ML teacher."
    )
])

human_prompt = ChatPromptTemplate.from_messages([
    (
        "human",
        "Explain {topic} in simple terms."
    )
])


# -------------------------------------------------
# Combine the prompts
# -------------------------------------------------

combined_prompt = system_prompt + human_prompt


# -------------------------------------------------
# Provide variable
# -------------------------------------------------

messages = combined_prompt.invoke({
    "topic": "Vector Embeddings"
})


# -------------------------------------------------
# Display Result
# -------------------------------------------------

print("=" * 60)
print("Prompt Composition")
print("=" * 60)

for message in messages.messages:

    print("\nType:")
    print(message.type)

    print("\nContent:")
    print(message.content)

    print("-" * 60)