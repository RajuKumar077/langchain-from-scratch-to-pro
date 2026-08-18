"""
Module 08 - Chains
Lesson 05 - Conditional Routing / Branching

Goal
----
- Dynamically route inputs to different chains based on custom logic or classification
- Understand conditional execution using RunnableLambda or RunnableBranch
- Build a topic-based routing pipeline
"""

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
from langchain_ollama import ChatOllama

# -------------------------------------------------
# Create Chat Model
# -------------------------------------------------

llm = ChatOllama(
    model="mistral:latest",
    temperature=0.3,
)

parser = StrOutputParser()


# -------------------------------------------------
# Step 1: Define Specialized Expert Chains
# -------------------------------------------------

# Specialist Chain 1: Tech/Code Expert
prompt_tech = ChatPromptTemplate.from_messages([
    ("system", "You are a Senior Software Engineer. Give a technical, code-oriented response in 2-3 sentences."),
    ("human", "{question}"),
])
tech_chain = prompt_tech | llm | parser

# Specialist Chain 2: Business/Finance Expert
prompt_business = ChatPromptTemplate.from_messages([
    ("system", "You are a Business Advisor. Give a strategic, market-focused response in 2-3 sentences."),
    ("human", "{question}"),
])
business_chain = prompt_business | llm | parser

# Specialist Chain 3: General Knowledge (Fallback)
prompt_general = ChatPromptTemplate.from_messages([
    ("system", "You are a friendly general assistant. Answer simply in 2-3 sentences."),
    ("human", "{question}"),
])
general_chain = prompt_general | llm | parser


# -------------------------------------------------
# Step 2: Classifier Chain
# -------------------------------------------------

prompt_classifier = ChatPromptTemplate.from_messages([
    (
        "system",
        "Classify the input question into exactly one category: 'tech', 'business', or 'general'. "
        "Return ONLY the single word classification."
    ),
    ("human", "{question}"),
])

classifier_chain = prompt_classifier | llm | parser


# -------------------------------------------------
# Step 3: Routing Function
# -------------------------------------------------

def route_question(info):
    """
    Examines the topic classification and returns the corresponding Runnable chain.
    """
    topic = info["topic"].strip().lower()

    if "tech" in topic:
        return tech_chain
    elif "business" in topic:
        return business_chain
    else:
        return general_chain


# -------------------------------------------------
# Step 4: Full Pipeline Assembly
# -------------------------------------------------

full_chain = (
    {
        "topic": classifier_chain,
        "question": lambda x: x["question"],
    }
    | RunnableLambda(route_question)
)


# -------------------------------------------------
# Run Demonstrations
# -------------------------------------------------

if __name__ == "__main__":
    test_questions = [
        "How do I fix a NullPointerException in Java?",
        "What is the average ROI for SaaS startup marketing?",
        "Why is the sky blue?",
    ]

    for question in test_questions:
        print("\n" + "=" * 60)
        print(f"QUESTION: {question}")
        print("=" * 60)

        # Inspect classification
        detected_topic = classifier_chain.invoke({"question": question})
        print(f"Detected Category: [{detected_topic.strip()}]")

        # Run routing chain
        response = full_chain.invoke({"question": question})
        print("\n--- RESPONSE ---")
        print(response)