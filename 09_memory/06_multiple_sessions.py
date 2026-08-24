"""
Module 09 - Memory
Lesson 06 - Multiple Sessions

Goal
----
- Manage multiple conversation sessions
- Dynamically create sessions
- Keep each session isolated
- Reuse one chain for all users
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

        print(
            f"\nCreating new session: {session_id}"
        )

        store[session_id] = (
            InMemoryChatMessageHistory()
        )

    return store[session_id]


# -------------------------------------------------
# Add Message History
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
# CREATE MULTIPLE SESSIONS
# =================================================

sessions = [
    "user_001",
    "user_002",
    "user_003",
]


# =================================================
# USER 1
# =================================================

print("\n" + "=" * 60)
print("USER 1")
print("=" * 60)

print(
    chat(
        "user_001",
        "My name is Raju and I am learning LangChain."
    )
)


# =================================================
# USER 2
# =================================================

print("\n" + "=" * 60)
print("USER 2")
print("=" * 60)

print(
    chat(
        "user_002",
        "My name is Amit and I am learning Python."
    )
)


# =================================================
# USER 3
# =================================================

print("\n" + "=" * 60)
print("USER 3")
print("=" * 60)

print(
    chat(
        "user_003",
        "My name is Priya and I am learning SQL."
    )
)


# =================================================
# ASK EACH USER ABOUT THEIR MEMORY
# =================================================

print("\n" + "=" * 60)
print("MEMORY TEST")
print("=" * 60)


for session_id in sessions:

    print(
        f"\n{session_id}:"
    )

    answer = chat(
        session_id,
        "What is my name and what am I learning?",
    )

    print("AI:", answer)


# =================================================
# DISPLAY ALL SESSIONS
# =================================================

print("\n" + "=" * 60)
print("ALL SESSION HISTORIES")
print("=" * 60)


for session_id, history in store.items():

    print(
        f"\n--- {session_id} ---"
    )

    for message in history.messages:

        print(
            f"{message.type}: "
            f"{message.content}"
        )