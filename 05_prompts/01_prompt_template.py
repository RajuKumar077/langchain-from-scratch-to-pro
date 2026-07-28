"""
Module 05 - Prompts
Lesson 01 - PromptTemplate

Goal:
- Understand what a PromptTemplate is
- Create a reusable prompt
- Pass variables into the prompt
"""

from langchain_core.prompts import PromptTemplate


# -------------------------------------------------
# Create Prompt Template
# -------------------------------------------------

prompt = PromptTemplate.from_template(
    "Explain {topic} in simple terms."
)


# -------------------------------------------------
# Provide a value for the variable
# -------------------------------------------------

final_prompt = prompt.invoke({
    "topic": "Machine Learning"
})


# -------------------------------------------------
# Print the result
# -------------------------------------------------

print("=" * 60)
print("Prompt Template")
print("=" * 60)

print(final_prompt)