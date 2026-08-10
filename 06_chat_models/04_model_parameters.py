"""
Module 06 - Chat Models
Lesson 04 - Model Parameters

Goal
----
- Understand model parameters
- Learn temperature
- Compare different model configurations
"""

from langchain_ollama import ChatOllama


# -------------------------------------------------
# Model with temperature = 0
# -------------------------------------------------

llm_deterministic = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Model with temperature = 0.8
# -------------------------------------------------

llm_creative = ChatOllama(
    model="mistral:latest",
    temperature=0.8
)

llm_medium = ChatOllama(
    model="mistral:latest",
    temperature=0.3
)
# -------------------------------------------------
# Question
# -------------------------------------------------

question = "Explain Python in one sentence."


# -------------------------------------------------
# Deterministic Response
# -------------------------------------------------

response_1 = llm_deterministic.invoke(question)


# -------------------------------------------------
# Creative Response
# -------------------------------------------------

response_2 = llm_creative.invoke(question)

response_3 = llm_medium.invoke(question)

# -------------------------------------------------
# Display Results
# -------------------------------------------------

print("=" * 60)
print("TEMPERATURE = 0")
print("=" * 60)

print(response_1.content)


print("\n" + "=" * 60)
print("TEMPERATURE = 0.8")
print("=" * 60)

print(response_2.content)

print("\n" + "=" * 60)
print("TEMPERATURE = 0.3")
print("=" * 60)
print(response_3.content)