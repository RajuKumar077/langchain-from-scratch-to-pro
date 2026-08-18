"""
Module 08 - Chains
Lesson 06 - Fallbacks & Error Handling

Goal
----
- Build resilient LCEL pipelines that gracefully handle model failures or invalid outputs
- Use the `.with_fallbacks()` method to switch to backup models or parsers automatically
- Handle API outages, rate limits, or bad formatting without crashing the application
"""

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama

# -------------------------------------------------
# 1. Primary Model (Prone to fail / strict syntax)
# -------------------------------------------------

# Primary model (e.g., using a smaller model that might fail JSON output)
primary_llm = ChatOllama(
    model="mistral:latest",
    temperature=0.0,
)

# Backup model (e.g., higher capability or alternative endpoint)
backup_llm = ChatOllama(
    model="llama3:latest",
    temperature=0.0,
)


# -------------------------------------------------
# 2. Strict JSON Parser & Prompt
# -------------------------------------------------

parser = JsonOutputParser()

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a structured data extractor. Return JSON with 'capital' and 'population' fields.\n{format_instructions}"
    ),
    (
        "human",
        "Extract data for country: {country}"
    ),
])


# -------------------------------------------------
# 3. Define Chains with Fallbacks
# -------------------------------------------------

# Primary chain that might fail if output format is invalid
primary_chain = prompt | primary_llm | parser

# Secondary backup chain
backup_chain = prompt | backup_llm | parser


# Chain with Fallback configuration
# If primary_chain raises any Exception, execution automatically routes to backup_chain
resilient_chain = primary_chain.with_fallbacks([backup_chain])


# -------------------------------------------------
# 4. Custom Fallback Sequence (Fallback to Default Dict)
# -------------------------------------------------

# You can also fall back to a simple static Runnable dictionary if all LLMs fail
from langchain_core.runnables import RunnableLambda

def default_fallback_response(input_data):
    return {
        "capital": "Unknown",
        "population": "Unavailable",
        "status": "Failed to fetch data from LLM"
    }

static_fallback_chain = RunnableLambda(default_fallback_response)

# Multi-tier resilience: Primary LLM -> Backup LLM -> Static Default Dictionary
ultra_resilient_chain = primary_chain.with_fallbacks([
    backup_chain,
    static_fallback_chain
])


# -------------------------------------------------
# Run Demonstrations
# -------------------------------------------------

if __name__ == "__main__":
    country = "France"

    print("=" * 60)
    print("RUNNING RESILIENT CHAIN WITH FALLBACKS...")
    print("=" * 60)

    try:
        # Executes primary chain; falls back automatically if parsing or model call fails
        result = ultra_resilient_chain.invoke({
            "country": country,
            "format_instructions": parser.get_format_instructions()
        })

        print("\n--- EXTRACTED DATA (DICT) ---")
        print(result)
        print(f"Capital: {result.get('capital')}")
        print(f"Population: {result.get('population')}")

    except Exception as e:
        print(f"\nChain completely failed: {e}")