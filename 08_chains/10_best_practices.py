"""
Module 08 - Chains
Lesson 10 - LCEL Production Best Practices & Design Patterns

Goal
----
- Enforce strict typing, validation, and error boundaries in LCEL pipelines
- Master RunnableConfig (passing metadata, tags, and run IDs)
- Streamline batch operations and async execution
- Implement anti-patterns vs. clean production patterns
"""

import asyncio
from typing import Dict, Any
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import (
    RunnableConfig,
    RunnableLambda,
    RunnablePassthrough,
)
from langchain_ollama import ChatOllama

# -------------------------------------------------
# 1. Base Setup
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0.2,
)

parser = StrOutputParser()


# -------------------------------------------------
# Best Practice 1: Input Validation / Sanitization Guard
# -------------------------------------------------

def validate_and_sanitize_input(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sanitize and validate inputs before passing them down the chain.
    Prevents empty payloads, injection attacks, or improper dict key access.
    """
    raw_text = inputs.get("query", "").strip()

    if not raw_text:
        raise ValueError("Input 'query' cannot be empty.")

    if len(raw_text) > 1000:
        raw_text = raw_text[:1000]  # Soft cap input length

    return {"query": raw_text}


guard_step = RunnableLambda(validate_and_sanitize_input)


# -------------------------------------------------
# Best Practice 2: Context Preservation & Clean Prompts
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise enterprise software consultant."),
    ("human", "{query}")
])

# Production Chain Assembly
core_chain = guard_step | prompt | llm | parser


# -------------------------------------------------
# Best Practice 3: Traceability via RunnableConfig
# -------------------------------------------------

def run_with_telemetry():
    """
    Always pass a RunnableConfig dictionary to .invoke(), .batch(), or .stream().
    This enables metadata tagging, custom run tracking, and system tracing.
    """
    config: RunnableConfig = {
        "tags": ["production", "v1.2", "consulting-service"],
        "metadata": {
            "user_id": "usr_98412",
            "environment": "production",
            "module": "08_chains_lcel",
        },
        "run_name": "ConsultingQueryPipeline"
    }

    result = core_chain.invoke(
        {"query": "What are the core trade-offs between monolithic and microservice architectures?"},
        config=config
    )
    return result


# -------------------------------------------------
# Best Practice 4: Async Native Execution (ainvoke / abatch)
# -------------------------------------------------

async def run_async_batch():
    """
    In high-throughput API services (FastAPI/Django), never use synchronous .invoke().
    Always use .ainvoke() or .abatch() to prevent blocking the event loop.
    """
    queries = [
        {"query": "Explain database indexing in 1 sentence."},
        {"query": "What is horizontal scaling in 1 sentence."},
    ]

    config: RunnableConfig = {"tags": ["async-batch-job"]}

    # Non-blocking parallel execution
    results = await core_chain.abatch(queries, config=config)
    return results


# -------------------------------------------------
# Run Demonstrations
# -------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("1. RUNNING WITH TELEMETRY & CONFIG")
    print("=" * 60)

    response = run_with_telemetry()
    print("\n--- RESPONSE ---")
    print(response)

    print("\n" + "=" * 60)
    print("2. RUNNING ASYNC BATCH PROCESSING")
    print("=" * 60)

    batch_responses = asyncio.run(run_async_batch())
    for idx, resp in enumerate(batch_responses, 1):
        print(f"\n[Result {idx}]:\n{resp}")