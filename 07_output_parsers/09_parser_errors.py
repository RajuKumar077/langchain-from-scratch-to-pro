"""
Module 07 - Output Parsers
Lesson 09 - Parser Errors

Goal
----
- Understand parsing failures
- Handle structured-output errors
- Prevent the application from crashing
"""

from pydantic import BaseModel, Field, ValidationError
from langchain_ollama import ChatOllama


# -------------------------------------------------
# Define Output Schema
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
    temperature=0,
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

Return:
- name
- role
- skills
"""


# -------------------------------------------------
# Call Model Safely
# -------------------------------------------------

try:

    response = structured_llm.invoke(question)

    print("=" * 60)
    print("STRUCTURED RESPONSE")
    print("=" * 60)

    print(response)

    print("\n" + "=" * 60)
    print("FIELDS")
    print("=" * 60)

    print("Name   :", response.name)
    print("Role   :", response.role)
    print("Skills :", response.skills)


except ValidationError as e:

    print("\n" + "=" * 60)
    print("VALIDATION ERROR")
    print("=" * 60)

    print("The model returned data that does not match")
    print("the expected Person structure.")

    print("\nDetails:")
    print(e)


except Exception as e:

    print("\n" + "=" * 60)
    print("MODEL / PARSER ERROR")
    print("=" * 60)

    print("Something went wrong while processing the response.")

    print("\nDetails:")
    print(e)