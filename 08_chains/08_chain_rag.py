"""
Module 08 - Chains
Lesson 08.A - RAG Pipeline with Passthrough State

Goal
----
- Build a Retrieval-Augmented Generation (RAG) pipeline in LCEL
- Use RunnablePassthrough.assign() to preserve input queries while attaching retrieved context
- Format context dynamically before passing to the prompt
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama

# -------------------------------------------------
# 1. Initialize LLM & Parser
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0.2,
)

parser = StrOutputParser()


# -------------------------------------------------
# 2. Mock Retriever Function
# -------------------------------------------------

def mock_vector_store_retriever(inputs: dict) -> str:
    """Simulates querying a vector database for relevant documentation."""
    query = inputs["question"].lower()

    knowledge_base = {
        "lcel": "LangChain Expression Language (LCEL) is a declarative way to compose chains using the pipe operator (|).",
        "passthrough": "RunnablePassthrough allows forwarding inputs unchanged or augmenting dictionary keys via assign().",
    }

    results = [text for key, text in knowledge_base.items() if key in query]
    return "\n".join(results) if results else "No relevant context found."


# -------------------------------------------------
# 3. Define Prompt Template
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are an assistant answering technical questions. "
        "Use ONLY the provided context to answer. If the context doesn't contain the answer, say 'I don't know'.\n\n"
        "Context:\n{context}"
    ),
    (
        "human",
        "Question: {question}"
    )
])


# -------------------------------------------------
# 4. Construct RAG Chain using Passthrough
# -------------------------------------------------

rag_chain = (
    # Retains {'question': '...'} and injects 'context' field automatically
    RunnablePassthrough.assign(context=mock_vector_store_retriever)
    | prompt
    | llm
    | parser
)


# -------------------------------------------------
# Execution
# -------------------------------------------------

if __name__ == "__main__":
    test_queries = [
        "What is LCEL?",
        "How does quantum computing work?"
    ]

    print("=" * 60)
    print("RUNNING RAG CHAIN WITH PASSTHROUGH")
    print("=" * 60)

    for q in test_queries:
        print(f"\n[QUERY]: {q}")
        response = rag_chain.invoke({"question": q})
        print(f"[ANSWER]: {response}\n" + "-" * 40)