"""
Module 07 - Output Parsers
Lesson 07 - Parser with LLM

Goal
----
- Connect an LLM with an output parser
- Understand the LLM output -> parser flow
- Use Pydantic for structured output
"""

from pydantic import BaseModel, Field

from langchain_ollama import ChatOllama


# -------------------------------------------------
# Define Output Schema
# -------------------------------------------------

class Movie(BaseModel):

    title: str = Field(
        description="The movie title"
    )

    genre: str = Field(
        description="The movie genre"
    )

    year: int = Field(
        description="The release year"
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

parser = llm.with_structured_output(Movie)


# -------------------------------------------------
# Input
# -------------------------------------------------

question = """
Give me information about the movie Inception.

Return:
- title
- genre
- year
"""


# -------------------------------------------------
# Run LLM + Parser
# -------------------------------------------------

try:

    response = parser.invoke(question)


    # -------------------------------------------------
    # Display Complete Response
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
    # Access Fields
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("MOVIE DETAILS")
    print("=" * 60)

    print("Title :", response.title)
    print("Genre :", response.genre)
    print("Year  :", response.year)


except Exception as e:

    print("\n" + "=" * 60)
    print("ERROR")
    print("=" * 60)

    print(e)