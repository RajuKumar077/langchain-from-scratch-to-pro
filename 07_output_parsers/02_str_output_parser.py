"""
Module 07 - Output Parsers
Lesson 02 - StrOutputParser

Goal
----
- Connect Prompt + Chat Model + Output Parser
- Convert AIMessage into a plain string
- Understand the | pipeline operator
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


# -------------------------------------------------
# Create Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI teacher. "
        "Explain concepts clearly and simply."
    ),
    (
        "human",
        "Explain {topic} in simple terms."
    )
])


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
# Create Pipeline
# -------------------------------------------------

chain = prompt | llm | parser


# -------------------------------------------------
# Get User Input
# -------------------------------------------------

topic = input("\nEnter a topic: ")


# -------------------------------------------------
# Run Complete Pipeline
# -------------------------------------------------

response = chain.invoke({
    "topic": topic
})


# -------------------------------------------------
# Display Result
# -------------------------------------------------

print("\n" + "=" * 60)
print("FINAL RESPONSE")
print("=" * 60)

print(response)

print("\nResponse Type:")
print(type(response))