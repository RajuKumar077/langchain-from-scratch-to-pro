"""
Lesson 06 - Custom LLM-as-a-Judge Evaluators with Structured Rubrics
===================================================================
LangSmith Principle:
  Rule-based evaluators (exact string match, keyword search, length heuristics)
  fail completely when evaluating natural language generation, summarization,
  and reasoning tasks.
  
  Production LLMOps relies on "LLM-as-a-Judge":
  1. A secondary LLM acts as an impartial grader.
  2. The Judge is provided with the Reference Question, Expected Answer, and Actual Prediction.
  3. The Judge follows a strict rubric, produces Chain-of-Thought reasoning, and emits
     a structured score (0.0 to 1.0) with an explanatory comment.
  4. LangSmith records both the student trace and the judge's evaluation side-by-side.

Run command:
  r:\\LangChain-Book\\.venv\\Scripts\\python.exe 06_llm_as_judge_evaluators.py
"""

import os
import json
import config
from langsmith import Client, evaluate
from langsmith.evaluation import EvaluationResult
from langsmith.schemas import Run, Example
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

DATASET_NAME = "LangSmith LLM-Judge QA Suite"

# ------------------------------------------------------------------------------
# 1. Dataset Setup (Question + Ground Truth Reference)
# ------------------------------------------------------------------------------
def get_or_create_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        print(f"[DATASET] Reusing existing dataset '{DATASET_NAME}'.")
        return client.read_dataset(dataset_name=DATASET_NAME)

    print(f"[DATASET] Creating benchmark dataset '{DATASET_NAME}'...")
    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Dataset for testing LLM-as-a-judge semantic evaluation."
    )

    benchmark_data = [
        {
            "inputs": {"question": "What is the primary difference between a LangChain Run and a Trace?"},
            "outputs": {"reference_answer": "A Run represents an individual component execution (e.g., a single LLM call, tool call, or chain step). A Trace is the complete hierarchical tree of nested runs forming an entire transaction."}
        },
        {
            "inputs": {"question": "Why should you use an Annotation Queue in LangSmith?"},
            "outputs": {"reference_answer": "Annotation queues allow human reviewers and domain experts to triage, inspect, and label low-confidence traces, production errors, or user feedback for quality assurance and fine-tuning."}
        }
    ]

    client.create_examples(
        inputs=[item["inputs"] for item in benchmark_data],
        outputs=[item["outputs"] for item in benchmark_data],
        dataset_id=dataset.id
    )
    print(f" -> Added {len(benchmark_data)} benchmark items with reference answers.")
    return dataset


# ------------------------------------------------------------------------------
# 2. Target Pipeline Being Evaluated (The "Student")
# ------------------------------------------------------------------------------
def target_student_pipeline(inputs: dict) -> dict:
    """The pipeline under test (using local Ollama model)."""
    question = inputs["question"]
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen3:8b"), temperature=0.2)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the technical question accurately and concisely in 2-3 sentences."),
        ("user", "{question}")
    ])
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question})
    return {"prediction": answer}


# ------------------------------------------------------------------------------
# 3. LLM-as-a-Judge Custom Evaluator
# ------------------------------------------------------------------------------
def llm_judge_factual_accuracy(run: Run, example: Example) -> EvaluationResult:
    """
    Custom LLM-as-a-Judge Evaluator.
    Uses local Ollama to grade the student's output against the reference answer.
    """
    question = example.inputs.get("question", "")
    reference = example.outputs.get("reference_answer", "")
    prediction = run.outputs.get("prediction", "") if run.outputs else ""

    # Instantiate Judge LLM (using low temperature for consistent grading)
    judge_llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen3:8b"), temperature=0.0)

    judge_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are a rigorous, objective LLMOps evaluator.\n"
            "Evaluate whether the STUDENT PREDICTION accurately conveys the core facts from the REFERENCE ANSWER.\n"
            "Format your response as valid JSON with keys:\n"
            "- 'reasoning': concise explanation of why the answer is accurate or inaccurate\n"
            "- 'score': 1.0 if factually correct and aligned with reference, else 0.0"
        )),
        ("user", (
            "QUESTION: {question}\n"
            "REFERENCE ANSWER: {reference}\n"
            "STUDENT PREDICTION: {prediction}\n\n"
            "Respond ONLY with a JSON object: {{\"reasoning\": \"...\", \"score\": 1.0 or 0.0}}"
        ))
    ])

    judge_chain = judge_prompt | judge_llm | StrOutputParser()
    judge_response = judge_chain.invoke({
        "question": question,
        "reference": reference,
        "prediction": prediction
    })

    # Parse judge verdict safely
    score = 0.0
    reasoning = "Parsing error"
    try:
        # Strip potential markdown code block wrappers
        clean_text = judge_response.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        data = json.loads(clean_text)
        score = float(data.get("score", 0.0))
        reasoning = data.get("reasoning", "No reasoning provided")
    except Exception:
        # Fallback keyword parsing if model returned raw text
        if "1.0" in judge_response or "score: 1" in judge_response.lower():
            score = 1.0
        reasoning = judge_response.strip()[:200]

    return EvaluationResult(
        key="llm_factual_accuracy",
        score=score,
        comment=reasoning
    )


# ------------------------------------------------------------------------------
# 4. Evaluation Runner
# ------------------------------------------------------------------------------
def main():
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("[ERROR] LANGCHAIN_API_KEY is required to run evaluations.")
        return

    client = Client()
    get_or_create_dataset(client)

    print("\n[EVALUATION] Launching LLM-as-a-Judge benchmark on LangSmith...")
    results = evaluate(
        target_student_pipeline,
        data=DATASET_NAME,
        evaluators=[llm_judge_factual_accuracy],
        experiment_prefix="local-qwen3-judge-experiment",
        metadata={
            "evaluator_type": "llm-as-a-judge",
            "student_model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            "judge_model": os.getenv("OLLAMA_MODEL", "qwen3:8b"),
            "rubric": "factual_accuracy_v1"
        }
    )

    print("\n[SUCCESS] LLM-as-a-Judge evaluation completed!")
    print(f" -> View detailed judge scorecards on LangSmith: https://smith.langchain.com -> Datasets -> '{DATASET_NAME}'")

if __name__ == "__main__":
    main()
