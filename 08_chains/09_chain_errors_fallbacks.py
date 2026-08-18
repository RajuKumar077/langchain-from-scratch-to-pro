"""
Module 08 - Chains
Lesson 08.B - Fault-Tolerant Chains with Fallbacks

Goal
----
- Build resilient LCEL pipelines that gracefully handle model or parsing failures
- Use .with_fallbacks() to switch between primary models, backup models, and static handlers
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

# -------------------------------------------------
# 1. Models & Parser
# -------------------------------------------------

primary_llm = ChatOllama(model="mistral:latest", temperature=0.0)
backup_llm = ChatOllama(model="llama3:latest", temperature=0.0)

parser = JsonOutputParser()

# -------------------------------------------------
# 2. Prompt Template requiring JSON
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Extract technical entities from the topic. Return JSON with 'topic' and 'key_concepts' fields.\n{format_instructions}"
    ),
    ("human", "Topic: {topic}")
])


# -------------------------------------------------
# 3. Primary & Secondary Chains
# -------------------------------------------------

primary_chain = prompt | primary_llm | parser
backup_chain = prompt | backup_llm | parser


# Static fallback handler if all LLMs fail
def safe_default_response(input_data: dict) -> dict:
    return {
        "topic": input_data.get("topic", "Unknown"),
        "key_concepts": ["Fallback response - Processing unavailable"],
        "status": "degraded"
    }

static_fallback_chain = RunnableLambda(safe_default_response)


# -------------------------------------------------
# 4. Assembling Resilient Chain
# -------------------------------------------------

resilient_chain = primary_chain.with_fallbacks([
    backup_chain,
    static_fallback_chain
])


# -------------------------------------------------
# Execution
# -------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("RUNNING RESILIENT FALLBACK CHAIN")
    print("=" * 60)

    payload = {
        "topic": "LangChain LCEL",
        "format_instructions": parser.get_format_instructions()
    }

    result = resilient_chain.invoke(payload)

    print("\n--- OUTPUT DICTIONARY ---")
    print(result)
    print(f"Data Type: {type(result)}")