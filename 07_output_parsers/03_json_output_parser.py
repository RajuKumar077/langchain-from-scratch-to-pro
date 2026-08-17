"""
Module 07 - Output Parsers
Lesson 03 - JSON Output Parser

Goal
----
- Understand JsonOutputParser
- Ask the LLM for structured JSON
- Parse the response into a Python dictionary
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_ollama import ChatOllama


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Create JSON Output Parser
# -------------------------------------------------

parser = JsonOutputParser()


# -------------------------------------------------
# Create Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful AI assistant.

Return the answer as valid JSON.

The JSON must contain exactly these fields:
- name
- role
- skills
"""
    ),
    (
        "human",
        "Create a profile for a fictional AI Engineer."
    )
])


# -------------------------------------------------
# Create Pipeline
# -------------------------------------------------

chain = prompt | llm | parser


# -------------------------------------------------
# Run Pipeline
# -------------------------------------------------

try:

    response = chain.invoke({})

    print("=" * 60)
    print("JSON RESPONSE")
    print("=" * 60)

    print(response)

    print("\nResponse Type:")
    print(type(response))


except Exception as e:

    print("\nError:")
    print(e)