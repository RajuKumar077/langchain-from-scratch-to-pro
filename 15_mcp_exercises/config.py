"""
Central Configuration & Environment Setup for MCP Exercises
===========================================================
This module ensures:
1. Windows PowerShell UTF-8 console output encoding.
2. Loading of .env variables into os.environ BEFORE LangChain/LangSmith imports.
3. Verification of Ollama local server connectivity and available models.
4. Absolute path resolution for MCP server scripts and Python virtual environment.
"""

import os
import sys
import urllib.request
import json
from pathlib import Path
from dotenv import load_dotenv

# Ensure UTF-8 output on Windows PowerShell
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Paths
MODULE_DIR = Path(__file__).parent.resolve()
PROJECT_ROOT = MODULE_DIR.parent.resolve()
SERVERS_DIR = MODULE_DIR / "servers"
PYTHON_EXECUTABLE = sys.executable

def setup_environment(verbose: bool = True) -> dict:
    """Load .env file, configure LangSmith, and verify local Ollama server."""
    # Find .env
    for env_path in [MODULE_DIR / ".env", PROJECT_ROOT / ".env", Path.cwd() / ".env"]:
        if env_path.exists():
            load_dotenv(env_path, override=True)
            break
    else:
        load_dotenv(override=True)

    # LangSmith configuration
    api_key = os.getenv("LANGCHAIN_API_KEY") or os.getenv("LANGSMITH_API_KEY")
    project_name = os.getenv("LANGCHAIN_PROJECT", "mcp-learning-exercises")
    tracing_enabled = os.getenv("LANGCHAIN_TRACING_V2", "false").lower() == "true"

    if api_key:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGCHAIN_API_KEY"] = api_key
        os.environ["LANGSMITH_API_KEY"] = api_key
        os.environ["LANGCHAIN_PROJECT"] = project_name
        os.environ["LANGCHAIN_ENDPOINT"] = os.getenv(
            "LANGCHAIN_ENDPOINT", "https://api.smith.langchain.com"
        )

    # Ollama settings
    ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model = os.getenv("OLLAMA_MODEL", "qwen3:8b")

    ollama_online = False
    available_models = []
    try:
        req = urllib.request.Request(f"{ollama_url}/api/tags")
        with urllib.request.urlopen(req, timeout=2) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                available_models = [m["name"] for m in data.get("models", [])]
                ollama_online = True
    except Exception:
        ollama_online = False

    config_info = {
        "python_exe": PYTHON_EXECUTABLE,
        "module_dir": str(MODULE_DIR),
        "servers_dir": str(SERVERS_DIR),
        "ollama_url": ollama_url,
        "ollama_model": ollama_model,
        "ollama_online": ollama_online,
        "available_models": available_models,
        "tracing_enabled": bool(api_key and tracing_enabled),
        "project_name": project_name,
    }

    if verbose:
        print("=" * 65)
        print(" [MCP CONFIG] MODEL CONTEXT PROTOCOL ENVIRONMENT CHECK")
        print("=" * 65)
        print(f" * Python Virtualenv   : {PYTHON_EXECUTABLE}")
        print(f" * Servers Directory   : {SERVERS_DIR}")
        print(f" * Ollama Status       : {'ONLINE' if ollama_online else 'OFFLINE (Start Ollama)'}")
        print(f" * Selected Model      : {ollama_model} ({'Installed' if any(ollama_model in m for m in available_models) else 'Not installed'})")
        print(f" * LangSmith Tracing   : {'ENABLED' if config_info['tracing_enabled'] else 'DISABLED'}")
        if config_info['tracing_enabled']:
            print(f" * LangSmith Project   : {project_name}")
        print("=" * 65)
        print()

    return config_info

if __name__ == "__main__":
    setup_environment(verbose=True)
