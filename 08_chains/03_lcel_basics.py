"""
Module 08 - Chains
Lesson 03 - LCEL Basics (Runnable Protocol)

Goal
----
- Learn the core mechanics of LangChain Expression Language (LCEL)
- Understand the Runnable interface methods: invoke(), batch(), and stream()
- Observe how LCEL components pass inputs and outputs seamlessly
"""

from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0.3,
)


# -------------------------------------------------
# Create Prompt Template
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a technical glossary assistant. Provide a 1-sentence definition."
    ),
    (
        "human",
        "Define the technical term: {term}"
    ),
])


# -------------------------------------------------
# Create Output Parser
# -------------------------------------------------

parser = StrOutputParser()


# -------------------------------------------------
# Build LCEL Chain
# -------------------------------------------------

chain = prompt | llm | parser


# -------------------------------------------------
# Run Demonstrations
# -------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------------
    # Method 1: .invoke() - Single Input Execution
    # ---------------------------------------------
    print("=" * 60)
    print("1. DEMO: .invoke()")
    print("=" * 60)

    term = "Vector Database"
    result = chain.invoke({"term": term})
    print(f"Term: {term}")
    print(f"Result: {result}\n")


    # ---------------------------------------------
    # Method 2: .batch() - Parallel Processing
    # ---------------------------------------------
    print("=" * 60)
    print("2. DEMO: .batch()")
    print("=" * 60)

    terms = [{"term": "Embeddings"}, {"term": "RAG"}, {"term": "Fine-tuning"}]
    results = chain.batch(terms)

    for item, res in zip(terms, results):
        print(f"[{item['term']}]: {res}")
    print()


    # ---------------------------------------------
    # Method 3: .stream() - Token-by-Token Response
    # ---------------------------------------------
    print("=" * 60)
    print("3. DEMO: .stream()")
    print("=" * 60)

    stream_term = "Quantization"
    print(f"Streaming answer for '{stream_term}': ", end="", flush=True)

    # Streams tokens real-time as mistral generates them
    for chunk in chain.stream({"term": stream_term}):
        print(chunk, end="", flush=True)

    print("\n")