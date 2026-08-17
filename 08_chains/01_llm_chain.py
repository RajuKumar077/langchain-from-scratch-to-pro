"""
Module 08 - Chains
Lesson 01 - Simple LLM Chain

Goal
----
- Understand basic chain execution in LangChain
- Combine Prompt, LLM, and Output Parser into a single chain using LCEL (|)
- Invoke the full pipeline with a single input dictionary
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0.7,
)


# -------------------------------------------------
# Create Prompt Template
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a concise AI assistant. Explain complex concepts in 2-3 clear sentences."
    ),
    (
        "human",
        "Explain the concept of {topic}."
    ),
])


# -------------------------------------------------
# Create Output Parser
# -------------------------------------------------

parser = StrOutputParser()


# -------------------------------------------------
# Build Chain (LCEL)
# -------------------------------------------------

# Piping Prompt -> LLM -> Output Parser
chain = prompt | llm | parser


# -------------------------------------------------
# Run Chain
# -------------------------------------------------

if __name__ == "__main__":
    topic = input("\nEnter a topic to explain: ")

    print("\n" + "=" * 60)
    print("RUNNING CHAIN...")
    print("=" * 60)

    # Invoking the chain automatically passes data through all 3 stages
    response = chain.invoke({"topic": topic})

    print("\n" + "=" * 60)
    print("CHAIN RESPONSE")
    print("=" * 60)
    print(response)