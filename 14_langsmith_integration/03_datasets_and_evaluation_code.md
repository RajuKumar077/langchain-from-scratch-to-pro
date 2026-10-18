
# 1. The core idea: What are we learning?

Imagine you've built an AI chatbot using your local Ollama model, `qwen3:8b`.

You ask it:

> What is LangSmith?

It answers:

> LangSmith is a platform used to monitor and evaluate LLM applications.

Great! But here's the problem:

**How do you know whether your AI is giving good answers consistently?**

You could manually ask it 100 questions and check every answer. But that would be tedious.

Instead, you can automate the process using **LangSmith Datasets and Evaluations**.

Here's what your code does:

![Image](https://images.openai.com/static-rsc-4/RYjyXsjmEsofcdUIsDhWWA6RgN2cbahuhxAJ_9olqYoWCXJeQVZd6nDHvYJacn1MTE3Y10Bpo3lM6CGFAClKxqBVrSbzJWGJgvgAZIruq7mXbu1a4H50qRSRXUha5GbRy4qPfTs_x51pW8EOEsFcNJ4v6YRAAtHchFUophw7hEZ6HJ4Oi5129evGQkzYkc6c?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/avjT8b2fQvuMCptld5_7Xj3tf36wt34HOcFguTlNHP_pVFVKoxt7Dk0bo3YWaY9TtIJHnXsoew5z_3kLpoO812bIyeoRnMfVXkcsJYZJtdmG-nZmRh-N8L9wMmRX3VvZ7EiDBIlw-mU_NnUAthRulK5iZdy2OaSeOuDTrOKl47Zvb1PmmKqq5UwN6dsCpT3O?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/jViA-zUjxJ0pdycFdxAOdmB-51QCsFSfkdaX2t_Ja_7Bra7w5MTQd3aRNZZY3Mv1sSTytDi9VSRd12itMRkOWVel3rvfjYorX-ZzkgdA3_K0uOIiKYWvC8GDtzcN9GGcYaxsWnWA2Ez9NV-wzrODH2bDyt29rflQrGDJOU-JwEQyqNfnWSGIvEmDtb76H13i?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/0WFWHgekMym0U5k_m40UnijZBVz_CjsvDw-cLck6RNK4_Gj0p78MacPDqdSqXnTfjkqVJ00v4NqxEtv4FxoyLZXWCuZ2Bec-KrVuP_78hFOlsCwzc3GjqAxXJkL6WMeA99lzlM_L_Obw44EIotbwE1f6ji-daAq4gk8a1pblhNM7EAuwESmLJcL-BBIVdvA4?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/YOMm6J7rSSqLKOygORtm_II3WWbO8aKDNJj36Wez2ClvpERWoTMIZpELNFc12-uLZHnDF3zaRjF69SCB4c0LE0BmtH12xqz06ElVTYJruL7wXHlLe-0gxV_v1KXHdXLRtQsrve2vr132R4WUnhttEkRQ5Jjpp2F7luR8jyJLeQgfp75IfmeNqUxrZmqPAbuG?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/BKErNIbEdQN58fm6FlsUxifyxJgrtGFOjTZPU1Kb_hQuVYj0OBrCoFBlzcUtOYk_XV_t65MCx7WoopLrfn29vSirbZJB8_VcwFHEZQHq2DIgfuoXSh8bBdnm-vgBoXEgA096UG5Wv0RMyyZi1EnMVJAlxK7n7CkTNf6-GCnePm3QnQXc5VrWSRXj_Rg12caq?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/OHq6MTN4b3IhYVMatND8aKSwYAP8rSP_MufKzh_ME9psI-FlN9iK3somWv3Xq-1NwdqD3AhY3NxYB_98PAAp7RARqLStajr2vaKPN91w8mYS2COQDlfmFJ7S_QAppGqoMjWIj9_45BlUpmc-3CALSMjr5zvZZo0hF7JaguPTuu5uyC7Ik8TYO5hdhDZ6PCM_?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/0YG9z6XFEJXqzxl3E2nrfQniJyMuv1y9dviT3TjffK6oXh7tdk1-qrWzc4EK6J5s8l-uu2Vw2KoxEuAELDM_tbZobXSJGYO2xIk0utnD4LLmfO-kYI-smjvCpjgqNml7UpwaynpfpjUlBqqQ8zWV66WYYl5AQe4w_qFvKm-J3MhdK56UiBZVuWuN-RMi0cv7?purpose=fullsize)

```text
       1. Prepare Questions
                |
                v
       2. Store Expected Answers
                |
                v
       3. Ask Local Ollama
                |
                v
       4. Evaluate AI Answers
                |
                v
       5. View Results in LangSmith
```

Let's understand the five core components.

| Component       | What it means                                                    |
| --------------- | ---------------------------------------------------------------- |
| Dataset         | A collection of test questions and expected answers              |
| Target function | The AI application you're testing                                |
| Evaluator       | A function that checks the AI's answers                          |
| `evaluate()`    | Runs your AI against the dataset and collects evaluation results |
| Experiment      | A record of one evaluation run, including predictions and scores |

**Remember this distinction:** Ollama generates the answers. LangSmith organizes the tests, records the results, and helps you evaluate the application.

---

# 2. Core concept #1: Dataset

Let's start with this part of your code.

```python
DATASET_NAME = "LangSmith QA Benchmark (Local Ollama)"
```

This is simply the name of your evaluation dataset.

Next, look at this example:

```python
examples = [
    {
        "inputs": {"question": "What is LangSmith?"},
        "outputs": {
            "expected": "A platform for building, testing, evaluating, and monitoring LLM applications."
        }
    }
]
```

There are two important things here.

### `inputs`: What we give the AI

```python
"inputs": {"question": "What is LangSmith?"}
```

This is the question that will be passed to your AI application.

### `outputs`: What we expect from the AI

```python
"outputs": {
    "expected": "A platform for building, testing, evaluating, and monitoring LLM applications."
}
```

This stores the expected answer, also called the **reference answer** or ground truth.

The model doesn't automatically receive this expected answer. It's stored for evaluation.

### Think of it like an exam.

| Question                            | Expected answer                                        |
| ----------------------------------- | ------------------------------------------------------ |
| What is LangSmith?                  | A platform for testing and monitoring LLM applications |
| What does tracing do?               | Records application execution details                  |
| How do you trace a Python function? | Use `@traceable`                                       |

You're creating an answer key before the exam.

Your AI will answer the questions independently, and the evaluator will assess its responses.

### How does the function work?

```python
def create_or_get_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        return client.read_dataset(dataset_name=DATASET_NAME)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Benchmark dataset for testing AI technical questions with local Ollama."
    )
```

Let's break it down.

**First:**

```python
client.has_dataset(dataset_name=DATASET_NAME)
```

Checks whether a dataset with that name already exists in LangSmith.

**If it exists:**

```python
client.read_dataset(dataset_name=DATASET_NAME)
```

Retrieves the existing dataset instead of creating another one.

**Otherwise:**

```python
client.create_dataset(...)
```

Creates a new dataset.

Finally, your code adds the test examples:

```python
client.create_examples(
    inputs=[ex["inputs"] for ex in examples],
    outputs=[ex["outputs"] for ex in examples],
    dataset_id=dataset.id
)
```

This extracts the questions and expected answers from your Python list and uploads them to the dataset.

**Why do this?** So you can reuse the same questions to evaluate your model repeatedly.

---

# 3. Core concept #2: Target function

Now let's look at the function that actually runs your AI.

```python
def target_qa_chain(inputs: dict) -> dict:
    question = inputs["question"]

    llm = ChatOllama(
        model="qwen3:8b",
        temperature=0.2
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer the user question accurately and concisely in 1-2 sentences."),
        ("user", "{question}")
    ])

    chain = prompt | llm | StrOutputParser()

    answer = chain.invoke({"question": question})

    return {"output": answer}
```

You already know some LangChain concepts, so focus on the function's role.

**This is the application being tested.**

Let's follow the execution.

### Step 1: Receive the question

```python
question = inputs["question"]
```

Suppose LangSmith supplies:

```python
inputs = {
    "question": "What is LangSmith?"
}
```

The function extracts the question.

### Step 2: Initialize the model

```python
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.2
)
```

This configures your local Ollama model.

* `model="qwen3:8b"` selects the model.
* `temperature=0.2` encourages relatively consistent, less random answers.

The model runs locally, assuming Ollama is running and the model is available.

### Step 3: Prepare the prompt

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user question accurately and concisely in 1-2 sentences."),
    ("user", "{question}")
])
```

The system message defines how the model should respond.

The user message contains the actual question.

### Step 4: Build the chain

```python
chain = prompt | llm | StrOutputParser()
```

This is the familiar LangChain pipeline:

```text
Question
   |
   v
Prompt Template
   |
   v
Qwen3:8b
   |
   v
String Output Parser
   |
   v
Plain-text Answer
```

`StrOutputParser()` extracts the model's response as a string.

### Step 5: Run the chain

```python
answer = chain.invoke({"question": question})
```

This executes the pipeline and obtains the answer.

### Step 6: Return the result

```python
return {"output": answer}
```

Why return a dictionary rather than just the answer?

Because your evaluation pipeline expects a structured result.

For example:

```python
{
    "output": "LangSmith is a platform for testing and monitoring LLM applications."
}
```

The `"output"` key is important because your evaluator functions retrieve the prediction using that key.

**Core takeaway:** `target_qa_chain()` is the function LangSmith calls to obtain a prediction for each dataset example.

---

# 4. Core concept #3: Evaluators

This is where the actual evaluation logic begins.

Your code defines two evaluators:

```python
evaluators=[
    evaluate_contains_keywords,
    evaluate_conciseness
]
```

The first checks keyword relevance. The second checks conciseness.

Let's understand why you need them.

Imagine the AI receives:

**Question:** What is LangSmith?

**Expected answer:**

> A platform for building, testing, evaluating, and monitoring LLM applications.

**AI prediction:**

> LangSmith helps developers monitor and evaluate their LLM applications.

Is the answer good?

Yes, it conveys the main idea, even though it doesn't match the expected answer word for word.

This is why evaluation matters: you need criteria to judge the response.

## Evaluator A: Keyword verification

Your code:

```python
def evaluate_contains_keywords(run, example) -> EvaluationResult:

    prediction = run.outputs.get("output", "") if run.outputs else ""
    expected = example.outputs.get("expected", "")

    score = 1.0 if any(
        word.lower() in prediction.lower()
        for word in expected.split()[:3]
    ) else 0.0

    return EvaluationResult(
        key="keyword_relevance",
        score=score,
        comment="Evaluated prediction against expected context."
    )
```

Let's understand the important variables first.

| Variable          | Meaning                                               |
| ----------------- | ----------------------------------------------------- |
| `run`             | The result of running your AI application             |
| `run.outputs`     | The output dictionary returned by the target function |
| `example`         | The original dataset example                          |
| `example.outputs` | The expected answer stored in the dataset             |
| `prediction`      | The actual answer generated by your AI                |
| `expected`        | The reference answer                                  |

### How does it retrieve the prediction?

```python
prediction = run.outputs.get("output", "") if run.outputs else ""
```

Remember the dictionary returned by your target function?

```python
{"output": answer}
```

This line extracts the answer from it.

If outputs are missing, it uses an empty string.

### How does it retrieve the expected answer?

```python
expected = example.outputs.get("expected", "")
```

This extracts the reference answer from the dataset.

### How does it calculate the score?

```python
score = 1.0 if any(
    word.lower() in prediction.lower()
    for word in expected.split()[:3]
) else 0.0
```

This is the key logic.

Suppose the expected answer is:

```text
A platform for building, testing, evaluating, and monitoring LLM applications.
```

This expression:

```python
expected.split()[:3]
```

takes the first three words:

```python
["A", "platform", "for"]
```

Then `any()` checks whether **at least one** of those words appears in the prediction, ignoring capitalization.

If one matches, the score is `1.0`. Otherwise, it is `0.0`.

In evaluation terms:

* `1.0` means the keyword check passed.
* `0.0` means the keyword check failed.

### ⚠️ There's a weakness in this evaluator.

Suppose the AI answers:

> A completely unrelated explanation.

The word `"A"` appears in that answer.

Your evaluator might give it a score of `1.0`, even though the answer is incorrect.

That's because the code checks whether *any one* of the first three words matches. It doesn't check whether the answer actually communicates the correct meaning.

For a learning exercise, this demonstrates how to create a custom evaluator. For a reliable benchmark, however, the logic needs improvement.

Also, the function's name is `evaluate_contains_keywords`, but it checks only the first three words of the expected answer, not a carefully selected set of important keywords.

---

## Evaluator B: Conciseness

Your second evaluator is simpler.

```python
def evaluate_conciseness(run, example) -> EvaluationResult:

    prediction = run.outputs.get("output", "") if run.outputs else ""

    word_count = len(prediction.split())

    is_concise = word_count <= 150

    return EvaluationResult(
        key="conciseness",
        score=1.0 if is_concise else 0.5,
        comment=f"Word count: {word_count} words."
    )
```

The goal is to check whether the answer contains 150 words or fewer.

### Step 1

```python
word_count = len(prediction.split())
```

Splits the answer into words and counts them.

For example:

```python
prediction = "LangSmith helps evaluate AI applications."
```

The word count is `5`.

### Step 2

```python
is_concise = word_count <= 150
```

If the answer contains at most 150 words, `is_concise` becomes `True`.

Otherwise, it becomes `False`.

### Step 3

```python
score=1.0 if is_concise else 0.5
```

The scoring rules are:

| Word count    | Score |
| ------------- | ----: |
| 150 or fewer  |   1.0 |
| More than 150 |   0.5 |

Notice that even a 500-word answer receives `0.5`, rather than `0.0`.

That is a scoring choice made by the author of this evaluator.

**One important detail:** This checks length, not quality. A short, incorrect answer could still receive a perfect conciseness score.

Also, your system prompt already asks for answers in one or two sentences. The conciseness evaluator checks whether the model follows a much looser limit of 150 words.

---

# 5. Core concept #4: What is `EvaluationResult`?

Both evaluator functions return an `EvaluationResult`.

For example:

```python
return EvaluationResult(
    key="conciseness",
    score=1.0,
    comment=f"Word count: {word_count} words."
)
```

Think of this as a structured evaluation record.

It contains three important pieces of information:

| Field     | Purpose                     |
| --------- | --------------------------- |
| `key`     | Names the metric            |
| `score`   | Stores the evaluation score |
| `comment` | Explains the result         |

For example:

```text
Metric: conciseness
Score: 1.0
Comment: Word count: 12 words.
```

LangSmith can record these evaluation results alongside the model's predictions.

**Important distinction:** `EvaluationResult` doesn't generate or modify the AI answer. It reports the evaluator's judgment of that answer.

---

# 6. Core concept #5: The `evaluate()` function

Now we reach the most important part of this lesson.

Inside `main()`:

```python
eval_results = evaluate(
    target_qa_chain,
    data=DATASET_NAME,
    evaluators=[
        evaluate_contains_keywords,
        evaluate_conciseness
    ],
    experiment_prefix="qwen3-8b-qa-experiment",
    metadata={
        "model": "qwen3:8b",
        "infrastructure": "local_ollama",
        "eval_version": "v1.0"
    }
)
```

This is where everything comes together.

Let's break down the arguments.

### `target_qa_chain`

```python
target_qa_chain
```

This is the function that generates answers.

LangSmith invokes it with the inputs from your dataset.

Notice that you pass the function itself, without parentheses. You're giving `evaluate()` a callable to execute.

### `data=DATASET_NAME`

```python
data=DATASET_NAME
```

This identifies the dataset containing the questions and expected answers.

LangSmith uses those examples as the test cases.

### `evaluators=[...]`

```python
evaluators=[
    evaluate_contains_keywords,
    evaluate_conciseness
]
```

These are the functions that assess the generated answers.

For each example, the evaluation pipeline runs your target application and applies the configured evaluators.

### `experiment_prefix`

```python
experiment_prefix="qwen3-8b-qa-experiment"
```

This provides a recognizable name prefix for the experiment.

For example, it helps you distinguish runs that evaluate your local Qwen model from runs that evaluate a different model or configuration.

### `metadata`

```python
metadata={
    "model": "qwen3:8b",
    "infrastructure": "local_ollama",
    "eval_version": "v1.0"
}
```

This stores additional information about the experiment.

It's useful when comparing different configurations later.

For instance, you might evaluate:

* Qwen3:8b with temperature `0.2`
* Qwen3:8b with temperature `0.7`
* Another local model with temperature `0.2`

Metadata helps you identify which configuration produced each experiment's results.

### What happens overall?

Suppose your dataset contains three questions.

Conceptually, the evaluation proceeds like this:

| Question                             | Target function     | Evaluators                  |
| ------------------------------------ | ------------------- | --------------------------- |
| What is LangSmith?                   | Generates an answer | Keyword check + conciseness |
| What does `LANGCHAIN_TRACING_V2` do? | Generates an answer | Keyword check + conciseness |
| How do you trace a Python function?  | Generates an answer | Keyword check + conciseness |

That means you get **six individual evaluator results** across the three examples: two scores per example.

LangSmith also records experiment-level information that helps you inspect the predictions and their evaluation results.

One important clarification: the scores from these two evaluators are separate metrics. They are not automatically combined into one overall accuracy score.

---

# 7. Core concept #6: Why do we need `main()`?

Your script has a `main()` function that coordinates the entire process.

```python
def main():
    api_key = os.getenv("LANGCHAIN_API_KEY")

    if not api_key:
        print("[ERROR] LANGCHAIN_API_KEY is required to run LangSmith dataset evaluations.")
        print("Please add your key in .env file before running this script.")
        return

    client = Client()

    create_or_get_dataset(client)

    eval_results = evaluate(
        target_qa_chain,
        data=DATASET_NAME,
        evaluators=[
            evaluate_contains_keywords,
            evaluate_conciseness
        ],
        experiment_prefix="qwen3-8b-qa-experiment",
        metadata={
            "model": "qwen3:8b",
            "infrastructure": "local_ollama",
            "eval_version": "v1.0"
        }
    )
```

The workflow is straightforward:

**1. Check the API key**

```python
api_key = os.getenv("LANGCHAIN_API_KEY")
```

Reads the environment variable containing your LangSmith API key.

The key is needed to communicate with the LangSmith service.

**2. Stop if the key is missing**

```python
if not api_key:
    return
```

Prevents the evaluation from proceeding without the required configuration.

**3. Initialize the client**

```python
client = Client()
```

Creates a LangSmith client to interact with the service.

**4. Create or retrieve the dataset**

```python
create_or_get_dataset(client)
```

Ensures that the dataset exists before evaluation.

**5. Execute the evaluation**

```python
eval_results = evaluate(...)
```

Runs the evaluation workflow.

At the end, the script prints a success message telling you to inspect the results in LangSmith.

The final block:

```python
if __name__ == "__main__":
    main()
```

ensures that `main()` runs when you execute this Python file directly.

---

# 8. One important distinction: Local Ollama vs. LangSmith

Since you're learning local AI development, this distinction matters.

| Task                                    | Where it happens                               |
| --------------------------------------- | ---------------------------------------------- |
| Running Qwen3:8b inference              | Your local Ollama environment                  |
| Constructing prompts                    | Your Python/LangChain code                     |
| Creating the evaluation dataset         | LangSmith service                              |
| Storing experiment records              | LangSmith service                              |
| Executing your custom Python evaluators | Your Python process, as part of the evaluation |
| Viewing experiment results              | LangSmith interface                            |

Your model can run locally without sending its inference requests to a hosted LLM provider.

However, **LangSmith is a remote service in this setup**. Dataset examples, predictions, and evaluation results can be uploaded to it as part of the evaluation workflow. Don't use confidential or sensitive data without checking your organization's policies and the service's data-handling requirements.

Also, your `setup_environment()` function comes from your own `config.py` file, so its exact behavior depends on how you implemented it.

---

# 9. The entire lesson in one picture

Let's connect everything.

```text
              LANGSMITH DATASET
             (Questions + Answers)
                      |
                      v
              target_qa_chain()
                      |
                      v
                 PromptTemplate
                      |
                      v
                 ChatOllama
                  Qwen3:8b
                      |
                      v
                 Generated Answer
                      |
              +-------+-------+
              |               |
              v               v
       Keyword Evaluator  Conciseness Evaluator
              |               |
              v               v
        Keyword Score     Length Score
              |               |
              +-------+-------+
                      |
                      v
                 LANGSMITH
              Experiment Results
```

The dataset provides the test cases, the target function generates answers, and the evaluators score those answers.

The `evaluate()` function coordinates this process.

That's the central idea behind this entire lesson.

---

## 10. What you should remember for interviews

Here are five questions worth being able to answer without looking at your code.

**Q1. What is a LangSmith dataset?**

A collection of evaluation examples containing inputs and, optionally, reference outputs used to test an LLM application.

**Q2. What is a target function?**

The application or callable being evaluated. In this script, `target_qa_chain()` generates answers using a local Qwen model through Ollama.

**Q3. What is a custom evaluator?**

A function that assesses a model's output against a defined criterion, such as keyword matching or response length.

**Q4. What does `evaluate()` do?**

It runs the target application against a dataset, executes the configured evaluators, and records the results as an experiment.

**Q5. Why use LangSmith when the model runs locally?**

Because local inference and experiment tracking are separate concerns. You can run the model locally while using LangSmith to organize datasets, compare experiments, and inspect evaluation results.

---
