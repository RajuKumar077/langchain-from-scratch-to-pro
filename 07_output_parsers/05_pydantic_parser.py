"""
Module 07 - Output Parsers
Lesson 05 - Pydantic Parser

Goal
----
- Understand Pydantic models
- Define a structured output schema
- Validate model output
- Use the schema with ChatOllama
"""

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama


# -------------------------------------------------
# Define Pydantic Schema
# -------------------------------------------------

class Person(BaseModel):

    name: str = Field(
        description="The person's name"
    )

    role: str = Field(
        description="The person's job role"
    )

    skills: list[str] = Field(
        description="A list of technical skills"
    )


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

Include the person's:
- name
- role
- technical skills
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
    print("PYDANTIC RESPONSE")
    print("=" * 60)

    print(response)


    # -------------------------------------------------
    # Display Type
    # -------------------------------------------------

    print("\nResponse Type:")
    print(type(response))


    # -------------------------------------------------
    # Access Fields
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("INDIVIDUAL FIELDS")
    print("=" * 60)

    print("Name   :", response.name)
    print("Role   :", response.role)
    print("Skills :", response.skills)


    # -------------------------------------------------
    # Convert to Dictionary
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("AS DICTIONARY")
    print("=" * 60)

    print(response.model_dump())


except Exception as e:

    print("\n" + "=" * 60)
    print("ERROR")
    print("=" * 60)

    print(e)