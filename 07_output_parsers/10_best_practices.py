"""
Module 07 - Output Parsers
Lesson 10 - Output Parser Best Practices

Goal
----
- Choose the right output format
- Use structured schemas when needed
- Validate structured output
- Handle errors
- Keep the LLM -> Parser pipeline clean
"""

from pydantic import BaseModel, Field, ValidationError

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# -------------------------------------------------
# Define Structured Output
# -------------------------------------------------

class Answer(BaseModel):

    answer: str = Field(
        description="The answer to the user's question"
    )

    confidence: str = Field(
        description="Confidence level: high, medium, or low"
    )


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# -------------------------------------------------
# Create Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.

        Answer the user's question clearly.

        Return:
        - answer
        - confidence

        Confidence must be one of:
        high, medium, low
        """
    ),
    (
        "human",
        "{question}"
    ),
])


# -------------------------------------------------
# Configure Structured Output
# -------------------------------------------------

structured_llm = llm.with_structured_output(Answer)


# -------------------------------------------------
# Create Pipeline
# -------------------------------------------------

chain = prompt | structured_llm


# -------------------------------------------------
# Get User Question
# -------------------------------------------------

question = input("\nAsk a Question: ")


# -------------------------------------------------
# Run Safely
# -------------------------------------------------

try:

    response = chain.invoke({
        "question": question
    })


    # -------------------------------------------------
    # Display Result
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("STRUCTURED ANSWER")
    print("=" * 60)

    print(response)


    # -------------------------------------------------
    # Access Fields
    # -------------------------------------------------

    print("\n" + "=" * 60)
    print("ANSWER")
    print("=" * 60)

    print(response.answer)

    print("\nConfidence:")
    print(response.confidence)


except ValidationError as e:

    print("\n" + "=" * 60)
    print("VALIDATION ERROR")
    print("=" * 60)

    print(e)


except Exception as e:

    print("\n" + "=" * 60)
    print("ERROR")
    print("=" * 60)

    print(e)