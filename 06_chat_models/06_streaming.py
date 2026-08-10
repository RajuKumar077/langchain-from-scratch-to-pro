"""
Module 06 - Chat Models
Lesson 06 - Streaming

Goal
----
- Understand streaming
- Receive the response progressively
- Compare streaming with normal invoke()
"""

from langchain_ollama import ChatOllama


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Question
# -------------------------------------------------

question = (
    "Explain Retrieval Augmented Generation "
    "in simple terms."
)


# -------------------------------------------------
# Streaming Response
# -------------------------------------------------

print("=" * 60)
print("STREAMING RESPONSE")
print("=" * 60)

for chunk in llm.stream(question):

    print(chunk.content, end="", flush=True)


print("\n")