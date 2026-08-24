"""
Module 09 - Memory
Lesson 04 - RunnableWithMessageHistory

Goal
----
- Connect memory with LangChain Runnables
- Use RunnableWithMessageHistory
- Understand session-based conversation memory
- Automatically manage previous messages
"""

from langchain_ollama import ChatOllama

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

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

        Use the conversation history to understand
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
# Create Basic Chain
# -------------------------------------------------

chain = prompt | llm


# -------------------------------------------------
# Store Sessions
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
# Function to Chat
# -------------------------------------------------

def chat(
    question: str,
    session_id: str,
):

    response = chain_with_history.invoke(
        {
            "question": question,
        },
        config={
            "configurable": {
                "session_id": session_id
            }
        },
    )

    return response.content


# -------------------------------------------------
# Start Conversation
# -------------------------------------------------

print("=" * 60)
print("MEMORY + RUNNABLE")
print("=" * 60)


session_id = "user_001"


# -------------------------------------------------
# Conversation 1
# -------------------------------------------------

answer = chat(
    "My name is Raju.",
    session_id,
)

print("\nAI:", answer)


# -------------------------------------------------
# Conversation 2
# -------------------------------------------------

answer = chat(
    "I am learning LangChain.",
    session_id,
)

print("\nAI:", answer)


# -------------------------------------------------
# Conversation 3
# -------------------------------------------------

answer = chat(
    "What am I learning?",
    session_id,
)

print("\nAI:", answer)


# -------------------------------------------------
# Conversation 4
# -------------------------------------------------

answer = chat(
    "What is my name?",
    session_id,
)

print("\nAI:", answer)


# -------------------------------------------------
# Display Session History
# -------------------------------------------------

print("\n" + "=" * 60)
print("SESSION HISTORY")
print("=" * 60)

history = get_session_history(session_id)

for message in history.messages:

    print(
        f"\n{message.type.upper()}:"
    )

    print(message.content)