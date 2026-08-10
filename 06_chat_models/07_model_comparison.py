"""
Module 06 - Chat Models
Lesson 07 - Model Comparison

Goal
----
- Understand that ChatOllama can use different models
- Use the same question with different models
- Compare their responses
"""

from langchain_ollama import ChatOllama


# -------------------------------------------------
# Models
# -------------------------------------------------

mistral = ChatOllama(
    model="mistral:latest",
    temperature=0
)

# If you have another model installed in Ollama,
# replace the name below with that model.
#
# Example:
# ollama list
#
# Then use the exact model name.

second_model = ChatOllama(
    model="mistral:latest",
    temperature=0.7
)


# -------------------------------------------------
# Question
# -------------------------------------------------

question = (
    "Explain Retrieval Augmented Generation "
    "in simple terms."
)


# -------------------------------------------------
# Mistral - Temperature 0
# -------------------------------------------------

print("=" * 60)
print("MODEL 1 - MISTRAL / TEMPERATURE 0")
print("=" * 60)

response_1 = mistral.invoke(question)

print(response_1.content)


# -------------------------------------------------
# Mistral - Temperature 0.7
# -------------------------------------------------

print("\n" + "=" * 60)
print("MODEL 2 - MISTRAL / TEMPERATURE 0.7")
print("=" * 60)

response_2 = second_model.invoke(question)

print(response_2.content)