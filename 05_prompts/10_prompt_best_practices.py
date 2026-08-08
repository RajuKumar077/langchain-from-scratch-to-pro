"""
Module 05 - Prompts
Lesson 10 - Prompt Best Practices

Goal
----
- Understand good prompt structure
- Separate instructions, context, and question
- Prevent unsupported answers
- Keep prompts clear and focused
"""

from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------
# Create a Well-Structured RAG Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Your task is to answer the user's question using
the provided context.

Rules:
1. Use only the provided context.
2. Do not invent information.
3. If the answer is not present in the context,
   clearly say that you don't know.
4. Keep the answer concise and easy to understand.

Context:
{context}
"""
    ),
    (
        "human",
        "Question: {question}"
    )
])


# -------------------------------------------------
# Example Context
# -------------------------------------------------

context = """
Raju works in the AI department.
His current project involves building
AI-powered applications using Python.
"""


# -------------------------------------------------
# Example Question
# -------------------------------------------------

question = "Which department does Raju work in?"


# -------------------------------------------------
# Create Final Prompt
# -------------------------------------------------

messages = prompt.invoke({
    "context": context,
    "question": question
})


# -------------------------------------------------
# Display Prompt
# -------------------------------------------------

print("=" * 60)
print("PROMPT BEST PRACTICES")
print("=" * 60)

for message in messages.messages:

    print(f"\n{message.type.upper()}:")
    print(message.content)