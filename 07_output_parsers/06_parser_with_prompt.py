"""
Module 07 - Output Parsers
Lesson 06 - Parser with Prompt

Goal
----
- Combine ChatPromptTemplate with structured output
- Use a Pydantic schema
- Create a Prompt -> LLM pipeline
- Receive structured Python data
"""

from pydantic import BaseModel, Field

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# -------------------------------------------------
# Define Output Structure
# -------------------------------------------------

class Person(BaseModel):

    name: str = Field(
        description="The person's name"
    )

    role: str = Field(
        description="The person's job role"
    )

    skills: list[str] = Field(
        description="The person's technical skills"
    )


# -------------------------------------------------
# Create Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are an AI assistant that extracts
        structured information from text.
        """
    ),
    (
        "human",
        """
        Create a fictional professional profile
        based on this request:

        {request}
        """
    ),
])


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# -------------------------------------------------
# Add Structured Output
# -------------------------------------------------

structured_llm = llm.with_structured_output(Person)


# -------------------------------------------------
# Create Pipeline
# -------------------------------------------------

chain = prompt | structured_llm


# -------------------------------------------------
# User Input
# -------------------------------------------------

request = input(
    "\nDescribe the professional you want: "
)


# -------------------------------------------------
# Run Pipeline
# -------------------------------------------------

try:

    response = chain.invoke({
        "request": request
    })


    # -------------------------------------------------
    # Display Result
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("STRUCTURED RESPONSE")
    print("=" * 60)

    print(response)


    # -------------------------------------------------
    # Display Fields
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("FIELDS")
    print("=" * 60)

    print("Name   :", response.name)
    print("Role   :", response.role)
    print("Skills :", response.skills)


except Exception as e:

    print("\n" + "=" * 60)
    print("ERROR")
    print("=" * 60)

    print(e)