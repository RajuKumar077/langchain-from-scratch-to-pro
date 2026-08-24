"""
Module 09 - Memory
Lesson 09 - Memory Errors

Goal
----
- Understand common conversation memory mistakes
- Handle missing session IDs safely
- Handle context window explosion (token limits)
- Implement basic error handling for memory workflows
"""

from langchain_ollama import ChatOllama

from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)

from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)

from langchain_core.runnables.history import (
    RunnableWithMessageHistory,
)


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0,
)


# -------------------------------------------------
# Create Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a helpful AI assistant.
        Use the conversation history context to answer questions.
        """
    ),

    MessagesPlaceholder(
        variable_name="history"
    ),

    (
        "human",
        "{question}"
    ),
])


# -------------------------------------------------
# Create Chain
# -------------------------------------------------

chain = prompt | llm


# -------------------------------------------------
# Session Store
# -------------------------------------------------

store = {}


# -------------------------------------------------
# Get Session History (With Error Guarding)
# -------------------------------------------------

def get_session_history(
    session_id: str,
) -> BaseChatMessageHistory:

    # ERROR PREVENTION 1: Guard against empty or invalid session IDs
    if not session_id or not isinstance(session_id, str):
        raise ValueError("Invalid session_id provided. It must be a non-empty string.")

    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()

    return store[session_id]


# -------------------------------------------------
# Add Memory to Chain
# -------------------------------------------------

chain_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="question",
    history_messages_key="history",
)


# -------------------------------------------------
# Safe Chat Function
# -------------------------------------------------

def safe_chat(
    session_id: str,
    question: str,
):

    try:
        response = chain_with_history.invoke(
            {
                "question": question,
            },
            config={
                "configurable": {
                    "session_id": session_id,
                }
            },
        )
        return response.content

    except ValueError as err:
        return f"[Session Error]: {err}"

    except Exception as err:
        # ERROR PREVENTION 2: Catch general execution/LLM failure safely
        return f"[Execution Error]: {err}"


# =================================================
# TEST 1: NORMAL EXECUTION
# =================================================

print("=" * 60)
print("TEST 1: NORMAL EXECUTION")
print("=" * 60)

session_id = "user_001"

answer = safe_chat(session_id, "My name is Raju.")
print("\nAI:", answer)

answer = safe_chat(session_id, "What is my name?")
print("\nAI:", answer)


# =================================================
# TEST 2: INVALID SESSION ID ERROR
# =================================================

print("\n" + "=" * 60)
print("TEST 2: INVALID SESSION ID HANDLER")
print("=" * 60)

# Passing an empty string session_id to trigger the guard
answer = safe_chat("", "Hello?")
print("\nResult:", answer)


# =================================================
# TEST 3: MANUAL HISTORY CORRUPTION HANDLING
# =================================================

print("\n" + "=" * 60)
print("TEST 3: CORRUPTED/OVERSIZED HISTORY GUARD")
print("=" * 60)

history = get_session_history(session_id)

# ERROR PREVENTION 3: Inspect and trim memory manually if history explodes
MAX_ALLOWED_MESSAGES = 10

if len(history.messages) > MAX_ALLOWED_MESSAGES:
    print("\n[Warning]: History exceeds limit. Trimming oldest messages...")
    # Keep only the last N messages to prevent context limit errors
    history.messages = history.messages[-MAX_ALLOWED_MESSAGES:]

print(f"Current total messages in store: {len(history.messages)}")