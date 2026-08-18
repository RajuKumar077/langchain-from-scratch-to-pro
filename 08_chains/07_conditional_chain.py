"""
Module 08 - Chains
Lesson 07 - RunnablePassthrough & Custom State Management

Goal
----
- Understand how to preserve original inputs while transforming or adding data in LCEL
- Use RunnablePassthrough to forward inputs untouched down the chain
- Build pipelines that carry raw user queries alongside generated context
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# -------------------------------------------------
# Create Chat Model & Parser
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0.3,
)

parser = StrOutputParser()


# -------------------------------------------------
# Step 1: Mock Context Fetcher (e.g., Vector DB / RAG)
# -------------------------------------------------

def mock_retriever(inputs):
    """Simulates fetching relevant background documents based on a query."""
    question = inputs["question"]
    if "python" in question.lower():
        return "Python is a high-level, interpreted programming language known for readability."
    return "No specific context found in database."


# -------------------------------------------------
# Step 2: Prompt Expecting Both Context and Question
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer the question using ONLY the provided context.\nContext: {context}"
    ),
    (
        "human",
        "Question: {question}"
    ),
])


# -------------------------------------------------
# Step 3: Build Chain using RunnablePassthrough
# -------------------------------------------------

# RunnablePassthrough() passes the incoming input object directly through untouched.
rag_chain = (
    {
        "context": mock_retriever,                       # Runs retriever function
        "question": RunnablePassthrough() | (lambda x: x["question"]),  # Passes raw question through
    }
    | prompt
    | llm
    | parser
)


# -------------------------------------------------
# Step 4: Alternative Pattern - RunnablePassthrough.assign()
# -------------------------------------------------

# Keeps existing keys in dictionary and dynamically adds/updates a new key ("context")
rag_chain_with_assign = (
    RunnablePassthrough.assign(context=mock_retriever)
    | prompt
    | llm
    | parser
)


# -------------------------------------------------
# Run Demonstrations
# -------------------------------------------------

if __name__ == "__main__":
    query = {"question": "What is Python?"}

    print("=" * 60)
    print("RUNNING PASSTHROUGH CHAIN...")
    print("=" * 60)

    # 1. Standard Passthrough execution
    response = rag_chain.invoke(query)

    print(f"\nUser Question: {query['question']}")
    print("\n--- RESPONSE ---")
    print(response)

    # 2. Inspecting state carried by .assign()
    print("\n" + "=" * 60)
    print("INSPECTING INTERMEDIATE STATE WITH .assign()...")
    print("=" * 60)

    state_builder = RunnablePassthrough.assign(context=mock_retriever)
    state_result = state_builder.invoke(query)

    print("\nCombined Dictionary Passed to Prompt:")
    print(state_result)
    # Output: {'question': 'What is Python?', 'context': 'Python is a high-level...'}