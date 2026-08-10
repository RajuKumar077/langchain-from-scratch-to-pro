"""
Module 06 - Chat Models
Lesson 05 - Temperature

Goal
----
- Understand the effect of temperature
- Compare low and high temperature
- Generate multiple responses
"""

from langchain_ollama import ChatOllama


# -------------------------------------------------
# Create Models
# -------------------------------------------------

low_temperature = ChatOllama(
    model="mistral:latest",
    temperature=0
)

high_temperature = ChatOllama(
    model="mistral:latest",
    temperature=0.9
)


# -------------------------------------------------
# Question
# -------------------------------------------------

question = (
    "Give me a creative one-line description "
    "of a rainy evening."
)


# -------------------------------------------------
# Temperature = 0
# -------------------------------------------------

print("=" * 60)
print("TEMPERATURE = 0")
print("=" * 60)

for i in range(3):

    response = low_temperature.invoke(question)

    print(f"\nResponse {i + 1}:")
    print(response.content)


# -------------------------------------------------
# Temperature = 0.9
# -------------------------------------------------

print("\n" + "=" * 60)
print("TEMPERATURE = 0.9")
print("=" * 60)

for i in range(3):

    response = high_temperature.invoke(question)

    print(f"\nResponse {i + 1}:")
    print(response.content)