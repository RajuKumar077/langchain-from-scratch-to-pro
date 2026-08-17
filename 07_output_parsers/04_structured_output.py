"""
Module 07 - Output Parsers
Lesson 04 - Structured Output

Goal
----
- Understand structured output
- Define the expected output schema
- Use ChatOllama.with_structured_output()
- Receive structured Python data
"""

from typing import TypedDict

from langchain_ollama import ChatOllama


# -------------------------------------------------
# Define Output Structure
# -------------------------------------------------

class Person(TypedDict):
    name: str
    role: str
    skills: list[str]


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Configure Structured Output
# -------------------------------------------------

structured_llm = llm.with_structured_output(Person)


# -------------------------------------------------
# Ask Question
# -------------------------------------------------

question = """
Create a fictional profile for an AI Engineer.

Include:
- name
- role
- skills
"""


# -------------------------------------------------
# Get Structured Response
# -------------------------------------------------

try:

    response = structured_llm.invoke(question)


    # -------------------------------------------------
    # Display Response
    # -------------------------------------------------

    print("=" * 60)
    print("STRUCTURED RESPONSE")
    print("=" * 60)

    print(response)


    # -------------------------------------------------
    # Display Type
    # -------------------------------------------------

    print("\nResponse Type:")
    print(type(response))


    # -------------------------------------------------
    # Access Individual Fields
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("INDIVIDUAL FIELDS")
    print("=" * 60)

    print("Name   :", response["name"])
    print("Role   :", response["role"])
    print("Skills :", response["skills"])


except Exception as e:

    print("\n" + "=" * 60)
    print("ERROR")
    print("=" * 60)

    print(e)