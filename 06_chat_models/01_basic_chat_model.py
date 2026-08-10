"""
Module 06 - Chat Models
Lesson 01 - Basic Chat Model

Goal
----
- Understand ChatOllama
- Send a simple message to the model
- Receive an AI response
"""

# pyrefly: ignore [missing-import]
from langchain_ollama import ChatOllama


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Send Message
# -------------------------------------------------

response = llm.invoke(
    "What is an embedding? Explain in simple terms."
)


# -------------------------------------------------
# Display Response
# -------------------------------------------------

print("=" * 60)
print("CHAT MODEL RESPONSE")
print("=" * 60)

print(response.content)