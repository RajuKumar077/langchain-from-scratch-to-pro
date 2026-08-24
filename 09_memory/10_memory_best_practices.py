"""
Module 09 - Memory
Lesson 10 - Memory Best Practices

Goal
----
- Implement production-ready memory patterns
- Automatically trim old messages (token management)
- Format clean session structures
- Keep prompts, history, and user input modular
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

from langchain_core.messages import trim_messages


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
        Answer questions concisely based on the conversation history.
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
# Memory Trimming Strategy (Best Practice)
# -------------------------------------------------

# Keep only the most recent 4 messages to avoid blowing up memory limits
trimmer = trim_messages(
    max_tokens=4,
    strategy="last",
    token_counter=len,  # Simple message count fallback
    include_system=False,
    start_on="human",
)


# -------------------------------------------------
# Create Chain with Trimmer
# -------------------------------------------------

# BEST PRACTICE: Pre-process history before sending to prompt
chain = prompt | llm


# -------------------------------------------------
# Session Store
# -------------------------------------------------

store = {}


# -------------------------------------------------
# Get Session History
# -------------------------------------------------

def get_session_history(
    session_id: str,
) -> BaseChatMessageHistory:

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
# Production Chat Function
# -------------------------------------------------

def chat(
    session_id: str,
    question: str,
):

    # BEST PRACTICE: Auto-trim history before execution
    history = get_session_history(session_id)
    if len(history.messages) > 6:
        # Keep only the last 4 messages in memory
        history.messages = history.messages[-4:]

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


# =================================================
# START CONVERSATION
# =================================================

print("=" * 60)
print("PRODUCTION MEMORY BEST PRACTICES")
print("=" * 60)

session_id = "prod_user_001"


# -------------------------------------------------
# Multi-Turn Simulation
# -------------------------------------------------

messages_to_send = [
    "My name is Raju.",
    "I live in Hyderabad.",
    "I am a Data Analyst.",
    "I am learning LangChain.",
    "What is my name?",
    "Where do I live?",
]

for idx, msg in enumerate(messages_to_send, 1):

    print(f"\n--- Turn {idx} ---")
    print(f"User: {msg}")

    answer = chat(session_id, msg)
    print(f"AI: {answer}")


# -------------------------------------------------
# Display Final History State
# -------------------------------------------------

print("\n" + "=" * 60)
print("FINAL TRIMMED SESSION HISTORY")
print("=" * 60)

history = get_session_history(session_id)

for message in history.messages:

    print(
        f"\n{message.type.upper()}: {message.content}"
    )