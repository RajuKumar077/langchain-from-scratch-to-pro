"""
Module 07 - Output Parsers
Lesson 01 - String Output Parser

Goal
----
- Understand what an LLM returns
- Extract the text from the AIMessage
- Understand StrOutputParser
"""

from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Create Output Parser
# -------------------------------------------------

parser = StrOutputParser()


# -------------------------------------------------
# Ask Question
# -------------------------------------------------

question = "What is a vector database?"


# -------------------------------------------------
# Get AI Response
# -------------------------------------------------

response = llm.invoke(question)


# -------------------------------------------------
# Display Raw Response
# -------------------------------------------------

print("=" * 60)
print("RAW AI RESPONSE")
print("=" * 60)

print(response)

print("\nResponse Type:")
print(type(response))


# -------------------------------------------------
# Parse Response
# -------------------------------------------------

parsed_response = parser.invoke(response)


# -------------------------------------------------
# Display Parsed Response
# -------------------------------------------------

print("\n" + "=" * 60)
print("PARSED RESPONSE")
print("=" * 60)

print(parsed_response)

print("\nParsed Type:")
print(type(parsed_response))