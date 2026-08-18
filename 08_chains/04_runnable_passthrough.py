"""
Module 08 - Chains
Lesson 04 - RunnableParallel

Goal
----
- Learn how to run multiple chains concurrently on the same input using LCEL
- Understand how RunnableParallel combines multiple outputs into a single dictionary
- Build a multi-perspective analysis pipeline
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_ollama import ChatOllama

# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0.5,
)

parser = StrOutputParser()


# -------------------------------------------------
# Step 1: Define Parallel Branch Chains
# -------------------------------------------------

# Branch A: Pros
prompt_pros = ChatPromptTemplate.from_messages([
    ("system", "List 2 major advantages or pros of the topic in concise bullet points."),
    ("human", "Topic: {topic}"),
])
chain_pros = prompt_pros | llm | parser

# Branch B: Cons
prompt_cons = ChatPromptTemplate.from_messages([
    ("system", "List 2 major disadvantages or cons of the topic in concise bullet points."),
    ("human", "Topic: {topic}"),
])
chain_cons = prompt_cons | llm | parser


# -------------------------------------------------
# Step 2: Build Parallel Chain (2 Ways to Write It)
# -------------------------------------------------

# Option A: Explicit RunnableParallel class
parallel_chain = RunnableParallel(
    pros=chain_pros,
    cons=chain_cons,
)

# Option B: Implicit dictionary syntax (equivalent in LCEL)
# parallel_chain = {
#     "pros": chain_pros,
#     "cons": chain_cons,
# }


# -------------------------------------------------
# Step 3: Synthesis Chain (Combines Parallel Outputs)
# -------------------------------------------------

prompt_summary = ChatPromptTemplate.from_messages([
    ("system", "You are a neutral analyst. Write a 2-sentence verdict balancing pros and cons."),
    ("human", "Pros:\n{pros}\n\nCons:\n{cons}\n\nProvide a final balanced verdict."),
])
summary_chain = prompt_summary | llm | parser


# -------------------------------------------------
# Full Sequential + Parallel Pipeline
# -------------------------------------------------

full_chain = parallel_chain | summary_chain


# -------------------------------------------------
# Run Demonstrations
# -------------------------------------------------

if __name__ == "__main__":
    topic = input("\nEnter a topic for analysis: ")

    print("\n" + "=" * 60)
    print("1. RUNNING PARALLEL BRANCHES ONLY...")
    print("=" * 60)

    # Invoking parallel_chain returns a dictionary containing keys 'pros' and 'cons'
    parallel_result = parallel_chain.invoke({"topic": topic})

    print("\n--- PROS ---")
    print(parallel_result["pros"])

    print("\n--- CONS ---")
    print(parallel_result["cons"])

    print("\n" + "=" * 60)
    print("2. RUNNING FULL PIPELINE (Parallel -> Synthesis)...")
    print("=" * 60)

    final_verdict = full_chain.invoke({"topic": topic})

    print("\n--- FINAL VERDICT ---")
    print(final_verdict)