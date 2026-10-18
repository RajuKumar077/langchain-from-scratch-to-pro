"""
Centralized Environment & LangSmith Initialization
===================================================
This module loads `.env` variables and ensures that LangSmith environment
variables are configured in `os.environ` BEFORE any LangChain/LangSmith imports.
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

def setup_environment():
    """Load environment variables from .env file and ensure LangSmith settings in os.environ."""
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    # Try loading .env from CWD, script folder, and project root
    current_dir_env = Path.cwd() / ".env"
    script_dir_env = Path(__file__).parent / ".env"
    root_dir_env = Path(__file__).parent.parent / ".env"

    loaded_path = None
    for env_path in [script_dir_env, root_dir_env, current_dir_env]:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            loaded_path = str(env_path)
            break

    if not loaded_path:
        dotenv_loaded = load_dotenv(override=True)
    else:
        dotenv_loaded = True

    # Explicitly ensure both V2 and new LANGSMITH env vars are populated in os.environ
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    project_name = os.getenv("LANGCHAIN_PROJECT", "default")
    
    if api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project_name
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv("LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com")

    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "").lower() == "true" or \
                      os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    
    print("=" * 60)
    print(" [CONFIG] LANGSMITH CONFIGURATION CHECK")
    print("=" * 60)
    print(f" * dotenv loaded        : {dotenv_loaded} ({loaded_path if loaded_path else 'No .env file found'})")
    print(f" * Tracing Enabled (V2) : {'YES' if tracing_enabled else 'NO (Set LANGCHAIN_TRACING_V2=true)'}")
    print(f" * LangSmith API Key    : {'PRESENT' if api_key else 'MISSING (Set LANGCHAIN_API_KEY)'}")
    print(f" * Target Project       : {project_name}")
    print("=" * 60)
    
    if not api_key:
        print("[NOTE] Without LANGCHAIN_API_KEY, runs execute locally but won't upload to LangSmith.")
    else:
        print(f"[SUCCESS] Traces will be uploaded to project '{project_name}' on https://smith.langchain.com")
    print()

# Run setup automatically when imported
setup_environment()
