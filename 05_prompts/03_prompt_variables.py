"""
Module 05 - Prompts
Lesson 03 - Prompt Variables

Goal:
- Understand multiple prompt variables
- Pass multiple values using a dictionary
- Understand how variables are replaced
"""

from langchain_core.prompts import PromptTemplate


# -------------------------------------------------
# Create Prompt Template
# -------------------------------------------------

prompt = PromptTemplate.from_template(
    "Explain {topic} for a {audience} using a {style} explanation."
)


# -------------------------------------------------
# Provide values for all variables
# -------------------------------------------------

final_prompt = prompt.invoke({
    "topic": "Retrieval Augmented Generation",
    "audience": "beginner",
    "style": "simple"
})


# -------------------------------------------------
# Print Result
# -------------------------------------------------

print("=" * 60)
print("Prompt Variables")
print("=" * 60)

print(final_prompt)