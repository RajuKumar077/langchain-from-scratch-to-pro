from pathlib import Path

content = """# Lesson 05: LangSmith Prompt Hub Management

## Goal
Learn how to store, reuse, and version-control prompts with LangSmith Prompt Hub.

## What the code does
1. **Push a prompt:** Creates a code-review prompt and saves it to LangSmith Hub as `code-reviewer-assistant`.
2. **Pull and execute:** Retrieves the prompt, connects it to the local Ollama model `qwen3:8b`, and runs a code review.
3. **Inspect history:** Lists the prompt's saved commits (versions).
4. **Trace execution:** LangSmith can record the chain run when tracing is enabled.

## Main functions

| Function | Purpose |
|---|---|
| `push_prompt_to_hub(client)` | Creates and publishes the prompt. |
| `pull_and_execute_prompt(client)` | Pulls the prompt and runs it with Ollama. |
| `inspect_prompt_commit_history(client)` | Displays the prompt's commit history. |
| `main()` | Runs the functions in order. |

## How the prompt works
The prompt template contains two messages:
- **System message:** Tells the AI to act as a senior software architect and review code.
- **User message:** Supplies the code snippet using `{code_snippet}`.

These two messages make up **one prompt template**, not two separate prompts.

## Execution flow
`Create prompt → Push to LangSmith Hub → Pull prompt → Send code to qwen3:8b → Print review → Inspect prompt history`

## How many scenarios are tested?
The current code tests **one scenario**:
- One Python function: `fetch_user_data(user_ids)`
- One `chain.invoke()` call
- One AI-generated code review

To test more scenarios, pass different code snippets to the chain.

## Key takeaway
**Create a prompt once, store it in LangSmith Hub, and reuse it in different applications.** Prompt versioning lets you track changes and select versions when needed.
"""
path = Path("/mnt/data/Lesson_05_LangSmith_Prompt_Hub.md")
path.write_text(content, encoding="utf-8")
print(f"Created: {path}")
