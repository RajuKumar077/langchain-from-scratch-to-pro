"""
Centralized Environment & LangSmith Initialization
===================================================
This module loads `.env` variables and verifies that LangSmith environment
variables are configured properly.
"""

import os
import sys
from dotenv import load_dotenv

def setup_environment():
    """Load environment variables from .env file and check LangSmith status."""
    # Load .env from current directory or root
    dotenv_loaded = load_dotenv()
    
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" or \
                      os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    
    print("=" * 60)
    print(" 🛠️  LANGSMITH CONFIGURATION CHECK")
    print("=" * 60)
    print(f" • dotenv loaded        : {dotenv_loaded}")
    print(f" • Tracing Enabled (V2) : {'✅ YES' if tracing_enabled else '❌ NO (Set LANGCHAIN_TRACING_V2=true)'}")
    print(f" • LangSmith API Key    : {'✅ PRESENT' if api_key else '⚠️ MISSING (Set LANGCHAIN_API_KEY)'}")
    print(f" • Target Project       : {project_name}")
    print("=" * 60)
    
    if not api_key:
        print("💡 NOTE: Without LANGCHAIN_API_KEY, runs will execute locally but won't be uploaded to LangSmith.")
    else:
        print("🚀 Traces will be uploaded to https://smith.langchain.com")
    print()

if __name__ == "__main__":
    setup_environment()
