"""
Module 09 - Memory
Lesson 07 - Memory with Prompt

Goal
----
- Combine ChatPromptTemplate with memory
- Use MessagesPlaceholder
- Keep system instructions separate from conversation history
- Understand how memory enters a prompt
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

    # System Instructions
    (
        "system",
        """
        You are a helpful AI assistant.

        Your job is to answer questions clearly
        and use the previous conversation when
        it is relevant.

        Important:
        - Do not invent information.
        - Keep answers concise.
        - Use the conversation history as context.
        """
    ),

    # Conversation Memory
    MessagesPlaceholder(
        variable_name="history"
    ),

    # Current User Question
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
# START CONVERSATION
# =================================================

print("=" * 60)
print("MEMORY + PROMPT")
print("=" * 60)


session_id = "user_001"


# -------------------------------------------------
# Turn 1
# -------------------------------------------------

print("\nUser:")
print("My name is Raju.")

answer = chat(
    session_id,
    "My name is Raju.",
)

print("\nAI:")
print(answer)


# -------------------------------------------------
# Turn 2
# -------------------------------------------------

print("\nUser:")
print("I am learning LangChain.")

answer = chat(
    session_id,
    "I am learning LangChain.",
)

print("\nAI:")
print(answer)


# -------------------------------------------------
# Turn 3
# -------------------------------------------------

print("\nUser:")
print("What am I learning?")

answer = chat(
    session_id,
    "What am I learning?",
)

print("\nAI:")
print(answer)


# -------------------------------------------------
# Turn 4
# -------------------------------------------------

print("\nUser:")
print("What is my name?")

answer = chat(
    session_id,
    "What is my name?",
)

print("\nAI:")
print(answer)


# -------------------------------------------------
# Display History
# -------------------------------------------------

print("\n" + "=" * 60)
print("STORED HISTORY")
print("=" * 60)

history = get_session_history(session_id)

for message in history.messages:

    print(
        f"\n{message.type.upper()}:"
    )

    print(message.content)