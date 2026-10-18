"""
Lesson 05 - LangSmith Prompt Hub Management & Version Control
============================================================
LangSmith Principle:
  In production, prompts are critical assets that evolve independently of
  application code. Hardcoding prompts in Python files requires code deployments
  for simple copy, instruction, or few-shot changes. 
  
  LangSmith Hub allows teams to:
  1. Push, version, and document prompt templates centrally.
  2. Pull prompts dynamically at runtime (hot-swapping prompts with zero downtime).
  3. Tag prompt commits (e.g., 'production', 'staging', 'canary') and rollback instantly.

Run command:
  r:\\LangChain-Book\\.venv\\Scripts\\python.exe 05_prompt_hub_management.py
"""

import os
import config
from langsmith import Client
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

PROMPT_HANDLE = "code-reviewer-assistant"

def push_prompt_to_hub(client: Client) -> str:
    """
    Step 1: Create a structured prompt template and push it to LangSmith Hub.
    In real workflows, prompts can also be created/edited directly in the LangSmith UI Playground.
    """
    print(f"\n[1. HUB PUSH] Defining and publishing prompt '{PROMPT_HANDLE}' to LangSmith Hub...")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a Senior Principal Software Architect conducting code reviews.\n"
            "Review the submitted code for:\n"
            "1. Potential bugs or edge cases\n"
            "2. Performance and time complexity\n"
            "3. Clean code and maintainability\n"
            "Provide concise, actionable recommendations."
        )),
        ("user", "Please review the following code snippet:\n```\n{code_snippet}\n```")
    ])

    # Push to LangSmith Hub with metadata & commit tag
    try:
        prompt_url = client.push_prompt(
            prompt_identifier=PROMPT_HANDLE,
            object=prompt,
            description="Senior engineer code review prompt with structured rubric.",
            tags=["code-review", "engineering-standards", "v1.0"],
            commit_description="Initial production commit of the code review prompt",
            is_public=False
        )
        print(f" -> Prompt successfully pushed to Hub!")
        print(f" -> View & test in Playground: {prompt_url}")
        return prompt_url
    except Exception as e:
        if "Nothing to commit" in str(e) or "Conflict" in str(e):
            print(f" -> [INFO] Prompt '{PROMPT_HANDLE}' already exists on Hub with identical content. Reusing existing commit.")
            return f"https://smith.langchain.com/prompts/{PROMPT_HANDLE}"
        raise e


def pull_and_execute_prompt(client: Client):
    """
    Step 2: Dynamically pull the latest prompt from LangSmith Hub and execute an LCEL chain.
    Notice that the application code has ZERO hardcoded prompt strings.
    """
    print(f"\n[2. HUB PULL] Fetching prompt '{PROMPT_HANDLE}' from LangSmith Hub...")
    pulled_prompt = client.pull_prompt(PROMPT_HANDLE)
    
    print(f" -> Successfully fetched prompt schema:")
    print(f"    Input Variables : {pulled_prompt.input_variables}")
    print(f"    Number of Steps : {len(pulled_prompt.messages)} messages")

    # Bind pulled prompt directly into LCEL pipeline
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen3:8b"), temperature=0.3)
    chain = pulled_prompt | llm | StrOutputParser()

    sample_code = (
        "def fetch_user_data(user_ids):\n"
        "    results = []\n"
        "    for uid in user_ids:\n"
        "        # Query database per user\n"
        "        data = db.query(f'SELECT * FROM users WHERE id = {uid}')\n"
        "        results.append(data)\n"
        "    return results"
    )

    print("\n[3. EXECUTION] Running LCEL chain with pulled Hub prompt...")
    response = chain.invoke(
        {"code_snippet": sample_code},
        config={
            "run_name": "Hub Prompt Code Review Run",
            "tags": ["hub-prompt", "code-review", "qwen3:8b"],
            "metadata": {"prompt_handle": PROMPT_HANDLE, "version": "latest"}
        }
    )

    print("\n--- Review Output (Generated via Hub Prompt & Local Ollama) ---")
    print(response)
    print("----------------------------------------------------------------\n")


def inspect_prompt_commit_history(client: Client):
    """
    Step 3: Inspect prompt version history programmatically.
    Enables audit trails and automated deployment gating.
    """
    print(f"[4. AUDIT] Listing commit history for prompt '{PROMPT_HANDLE}'...")
    try:
        commits = list(client.list_prompt_commits(PROMPT_HANDLE))
        print(f" -> Found {len(commits)} commit(s):")
        for c in commits:
            commit_hash = getattr(c, "commit_hash", str(c))
            description = getattr(c, "description", "No description")
            print(f"    - Commit [{commit_hash[:8]}]: {description}")
    except Exception as e:
        print(f"    [NOTE] Commit inspection: {e}")


def main():
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("[ERROR] LANGCHAIN_API_KEY is required to interact with LangSmith Prompt Hub.")
        return

    client = Client()
    push_prompt_to_hub(client)
    pull_and_execute_prompt(client)
    inspect_prompt_commit_history(client)
    
    # Flush traces to LangSmith
    client.flush()
    print("[SUCCESS] Prompt Hub lifecycle completed successfully!")

if __name__ == "__main__":
    main()
