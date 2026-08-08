"""
Module 05 - Prompts
Lesson 06 - Prompt + LLM

Goal:
- Create a ChatPromptTemplate
- Connect it to Ollama
- Send the prompt to the LLM
- Receive an AI response
"""

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama


# -------------------------------------------------
# Create Prompt
# -------------------------------------------------

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI/ML teacher. "
        "Explain technical concepts simply."
    ),
    (
        "human",
        "Explain {topic} in simple terms."
    )
])


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Create Prompt
# -------------------------------------------------

messages = prompt.invoke({
    "topic": "Vector Embeddings"
})


print("=" * 60)
print("Prompt")
print("=" * 60)

for message in messages.messages:
    print(f"{message.type}: {message.content}")


# -------------------------------------------------
# Send Prompt to LLM
# -------------------------------------------------

response = llm.invoke(messages)


# -------------------------------------------------
# Display Response
# -------------------------------------------------

print("\n" + "=" * 60)
print("LLM Response")
print("=" * 60)

print(response.content)