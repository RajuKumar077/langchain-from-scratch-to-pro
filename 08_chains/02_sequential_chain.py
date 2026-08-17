"""
Module 08 - Chains
Lesson 02 - Sequential Chain

Goal
----
- Learn how to pass the output of one chain as the input to a second chain
- Construct a multi-step workflow using LCEL (|)
- Generate an outline first, then expand it into a concise summary
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0.5,
)


# -------------------------------------------------
# Parser
# -------------------------------------------------

parser = StrOutputParser()


# -------------------------------------------------
# Step 1: Create Outline Chain
# -------------------------------------------------

prompt_outline = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an expert content creator. Create 3 main bullet points outlining a topic."
    ),
    (
        "human",
        "Create an outline for the topic: {topic}"
    ),
])

outline_chain = prompt_outline | llm | parser


# -------------------------------------------------
# Step 2: Create Summary Chain
# -------------------------------------------------

prompt_summary = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a tech writer. Take the provided outline and write a concise 1-paragraph summary."
    ),
    (
        "human",
        "Outline:\n{outline}\n\nWrite a short summary based on this outline."
    ),
])

summary_chain = prompt_summary | llm | parser


# -------------------------------------------------
# Full Sequential Chain (LCEL Composition)
# -------------------------------------------------

# Passes output of outline_chain directly as input {"outline": ...} to summary_chain
full_chain = (
    {"outline": outline_chain}
    | summary_chain
)


# -------------------------------------------------
# Run Sequential Chain
# -------------------------------------------------

if __name__ == "__main__":
    topic = input("\nEnter a topic for sequential generation: ")

    print("\n" + "=" * 60)
    print("RUNNING SEQUENTIAL CHAIN...")
    print("=" * 60)

    # 1. Inspect intermediate output (Step 1 alone)
    outline = outline_chain.invoke({"topic": topic})

    print("\n--- Intermediate Output (Outline) ---")
    print(outline)

    # 2. Run full pipeline (Step 1 -> Step 2)
    final_summary = full_chain.invoke({"topic": topic})

    print("\n" + "=" * 60)
    print("FINAL SUMMARY")
    print("=" * 60)
    print(final_summary)