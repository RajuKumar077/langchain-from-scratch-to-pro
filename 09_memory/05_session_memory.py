"""
Module 09 - Memory
Lesson 05 - Session Memory

Goal
----
- Understand session-based memory
- Create multiple independent sessions
- Keep conversations isolated
- Reuse the same chain for different users
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

        Use the conversation history to answer
        the user's current question.
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
# Get Session History
# -------------------------------------------------

def get_session_history(
    session_id: str,
) -> BaseChatMessageHistory:

    if session_id not in store:

        store[session_id] = (
            InMemoryChatMessageHistory()
        )

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
# Chat Function
# -------------------------------------------------

def chat(
    session_id: str,
    question: str,
):

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
# SESSION 1
# =================================================

print("\n" + "=" * 60)
print("SESSION 1")
print("=" * 60)

session_1 = "user_001"


print("\nUser 1:")
print("My name is Raju.")

answer = chat(
    session_1,
    "My name is Raju.",
)

print("\nAI:", answer)


print("\nUser 1:")
print("I am learning LangChain.")

answer = chat(
    session_1,
    "I am learning LangChain.",
)

print("\nAI:", answer)


print("\nUser 1:")
print("What am I learning?")

answer = chat(
    session_1,
    "What am I learning?",
)

print("\nAI:", answer)


# =================================================
# SESSION 2
# =================================================

print("\n" + "=" * 60)
print("SESSION 2")
print("=" * 60)

session_2 = "user_002"


print("\nUser 2:")
print("My name is Amit.")

answer = chat(
    session_2,
    "My name is Amit.",
)

print("\nAI:", answer)


print("\nUser 2:")
print("I am learning Python.")

answer = chat(
    session_2,
    "I am learning Python.",
)

print("\nAI:", answer)


print("\nUser 2:")
print("What am I learning?")

answer = chat(
    session_2,
    "What am I learning?",
)

print("\nAI:", answer)


# =================================================
# TEST SESSION ISOLATION
# =================================================

print("\n" + "=" * 60)
print("SESSION ISOLATION TEST")
print("=" * 60)


print("\nAsk User 1:")
answer = chat(
    session_1,
    "What is my name?",
)

print("AI:", answer)


print("\nAsk User 2:")
answer = chat(
    session_2,
    "What is my name?",
)

print("AI:", answer)


# =================================================
# DISPLAY STORED SESSIONS
# =================================================

print("\n" + "=" * 60)
print("AVAILABLE SESSIONS")
print("=" * 60)

for session_id in store:

    print(
        f"\nSession: {session_id}"
    )

    history = store[session_id]

    for message in history.messages:

        print(
            f"{message.type}: "
            f"{message.content}"
        )