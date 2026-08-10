"""
Module 06 - Chat Models
Lesson 09 - Error Handling

Goal
----
- Handle errors from the chat model
- Prevent the application from crashing
- Give the user a useful error message
"""

from langchain_ollama import ChatOllama


# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0
)


# -------------------------------------------------
# Get User Question
# -------------------------------------------------

question = input("\nAsk a Question: ")


# -------------------------------------------------
# Call Model Safely
# -------------------------------------------------

try:

    response = llm.invoke(question)

    print("\n" + "=" * 60)
    print("AI RESPONSE")
    print("=" * 60)

    print(response.content)


except Exception as e:

    print("\n" + "=" * 60)
    print("ERROR")
    print("=" * 60)

    print("Something went wrong while calling the model.")
    print(f"\nDetails: {e}")