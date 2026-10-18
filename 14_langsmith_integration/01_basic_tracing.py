"""
Lesson 01 - Basic LangSmith Tracing with Local Ollama
=====================================================
Goal:
  1. Use your LOCAL Ollama LLM (`qwen3:8b`) with zero OpenAI/Anthropic dependencies.
  2. Ensure LangSmith environment variables are set BEFORE importing LangChain.
  3. Flush trace batches so short scripts upload to LangSmith before exiting.

Run command:
  r:\\LangChain-Book\\.venv\\Scripts\\python.exe 01_basic_tracing.py
"""

# CRITICAL: Import config FIRST to set os.environ before LangChain initializes!
import config

import sys
import os
from langsmith import traceable, Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

# ------------------------------------------------------------------------------
# 1. Custom Python Function Tracing with @traceable
# ------------------------------------------------------------------------------
@traceable(
    name="Preprocess User Input",
    run_type="tool",
    tags=["preprocessing", "validation"]
)
def clean_user_input(text: str) -> str:
    """A custom helper function that gets traced in LangSmith UI."""
    cleaned = text.strip()
    if len(cleaned) < 3:
        raise ValueError("Input text is too short!")
    return f"[VERIFIED REQUEST]: {cleaned}"


# ------------------------------------------------------------------------------
# 2. Main Traced Pipeline
# ------------------------------------------------------------------------------
def run_traced_pipeline():
    user_query = "What are the core benefits of using LangSmith with local Ollama models?"

    # Step A: Run custom function wrapped in @traceable
    processed_input = clean_user_input(user_query)

    # Step B: Build LangChain LCEL Chain with local Ollama (qwen3:8b)
    llm = ChatOllama(
        model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        temperature=0.7
    )
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert AI software engineer. Explain concepts concisely."),
        ("user", "{input}")
    ])
    
    chain = prompt | llm | StrOutputParser()

    # Step C: Invoke chain with metadata & tags
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    print(f"[PROCESSING] Executing LCEL chain with local Ollama (qwen3:8b)...")
    response = chain.invoke(
        {"input": processed_input},
        config={
            "run_name": "LangSmith Local Ollama Query",
            "tags": ["demo", "chapter14", "local-ollama", "qwen3:8b"],
            "metadata": {
                "environment": "local_development",
                "infrastructure": "ollama_docker",
                "model_name": "qwen3:8b"
            }
        }
    )

    print("\n--- Model Response (Local Ollama qwen3:8b) ---")
    print(response)
    print("----------------------------------------------\n")
    
    # Step D: Flush background tracers to guarantee delivery to LangSmith
    print("[FLUSHING TRACES] Synchronizing trace logs with LangSmith...")
    try:
        client = Client()
        client.flush()
    except Exception as e:
        print(f"[NOTE] Trace flush warning: {e}")

    print(f"[SUCCESS] Trace sent! Open https://smith.langchain.com -> Project '{project_name}' to view.")

if __name__ == "__main__":
    run_traced_pipeline()
