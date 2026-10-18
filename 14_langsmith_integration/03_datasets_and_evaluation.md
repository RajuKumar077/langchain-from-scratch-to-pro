# Lesson 03: LangSmith Datasets & Automated Evaluations with Local Ollama

## 1. What Is the Main Goal of This Lesson?

We are learning how to evaluate an AI application automatically instead of manually checking every answer.

Our setup consists of:

* **LangChain:** Builds the LLM application pipeline.
* **Ollama:** Runs our local Qwen3:8b model.
* **LangSmith:** Stores evaluation datasets, tracks experiments, and displays evaluation results.
* **Custom evaluators:** Check whether the AI answers satisfy our chosen criteria.

### The overall workflow

Dataset → Target Function → Local LLM → Generated Answer → Evaluators → Evaluation Results

The central idea is simple:

We prepare questions with expected answers, run our AI application against those questions, and evaluate the resulting predictions.

---

## 2. Understanding the Dataset

### Dataset name

```python
DATASET_NAME = "LangSmith QA Benchmark (Local Ollama)"
```

`DATASET_NAME` is a custom variable created by the developer.

It stores the name of our evaluation dataset.

The dataset itself is managed through LangSmith.

### What is a dataset?

A dataset is a collection of examples used to test an application.

Our examples have two important components:

```python
{
    "inputs": {
        "question": "What is LangSmith?"
    },
    "outputs": {
        "expected": "A platform for building, testing, evaluating, and monitoring LLM applications."
    }
}
```

**Inputs:** The question given to our AI application.

**Outputs:** The expected answer used as a reference during evaluation.

The expected answer is often called the reference answer or ground truth.

### Why do we need expected answers?

Suppose our AI receives the question:

"What is LangSmith?"

It generates:

"LangSmith helps developers monitor and evaluate LLM applications."

We can compare that prediction against the reference answer to assess whether the response meets our evaluation criteria.

The expected answer is not automatically supplied to the model. It is stored in the dataset for evaluation.

### Creating or retrieving a dataset

```python
def create_or_get_dataset(client: Client):
    if client.has_dataset(dataset_name=DATASET_NAME):
        return client.read_dataset(dataset_name=DATASET_NAME)

    dataset = client.create_dataset(
        dataset_name=DATASET_NAME,
        description="Benchmark dataset for testing AI technical questions with local Ollama."
    )
```

This function checks whether the dataset already exists.

If it exists, the function retrieves it.

Otherwise, it creates a new dataset.

The examples are subsequently added using:

```python
client.create_examples(
    inputs=[ex["inputs"] for ex in examples],
    outputs=[ex["outputs"] for ex in examples],
    dataset_id=dataset.id
)
```

The purpose is to create a reusable collection of test cases.

---

## 3. Understanding the Target Function

Our target function is:

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

### What does the target function do?

It receives a question, sends it through a LangChain pipeline, obtains an answer from our local model, and returns the prediction.

### Step 1: Receive the input

```python
question = inputs["question"]
```

This extracts the question from the input dictionary.

For example:

```python
inputs = {
    "question": "What is LangSmith?"
}
```

### Step 2: Initialize the model

```python
llm = ChatOllama(
    model="qwen3:8b",
    temperature=0.2
)
```

`ChatOllama` is provided by the LangChain Ollama integration.

`qwen3:8b` identifies the model selected for inference.

The temperature setting influences response variability.

### Step 3: Create the prompt

```python
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the user question accurately and concisely in 1-2 sentences."),
    ("user", "{question}")
])
```

The system message defines the response instructions.

The user message contains the question.

### Step 4: Build the chain

```python
chain = prompt | llm | StrOutputParser()
```

This constructs a pipeline:

Prompt Template → Ollama Model → String Output Parser

The `|` operator connects the components.

### Step 5: Execute the chain

```python
answer = chain.invoke({"question": question})
```

This executes the pipeline and obtains the model's answer.

### Step 6: Return the prediction

```python
return {"output": answer}
```

The function returns a dictionary containing the generated answer.

This structure is important because our evaluators retrieve the prediction using the `"output"` key.

---

## 4. Predefined Names vs. Custom Names

An important skill when reading Python code is recognizing where each name comes from.

There are three main categories.

### A. Python keywords and built-ins

These are provided by Python itself.

| Name          | Category                | Purpose                                       |
| ------------- | ----------------------- | --------------------------------------------- |
| `def`         | Keyword                 | Defines a function                            |
| `if`          | Keyword                 | Checks a condition                            |
| `else`        | Keyword                 | Handles an alternative condition              |
| `return`      | Keyword                 | Returns a value                               |
| `import`      | Keyword                 | Imports a module or its contents              |
| `from`        | Keyword                 | Specifies an import source                    |
| `for`         | Keyword                 | Iterates over items                           |
| `in`          | Keyword/operator        | Checks membership or iterates over a sequence |
| `is`          | Keyword/operator        | Tests object identity                         |
| `True`        | Built-in constant       | Represents true                               |
| `False`       | Built-in constant       | Represents false                              |
| `None`        | Built-in constant       | Represents the absence of a value             |
| `print()`     | Built-in function       | Displays information                          |
| `len()`       | Built-in function       | Returns the length of an object               |
| `any()`       | Built-in function       | Returns true if at least one item is true     |
| `os.getenv()` | Standard-library method | Reads an environment variable                 |

### B. Library-provided names

These come from external libraries or modules.

| Name                 | Source                       | Purpose                               |
| -------------------- | ---------------------------- | ------------------------------------- |
| `Client`             | LangSmith                    | Interacts with the LangSmith service  |
| `evaluate()`         | LangSmith                    | Runs an evaluation experiment         |
| `EvaluationResult`   | LangSmith                    | Represents an evaluation result       |
| `ChatPromptTemplate` | LangChain                    | Creates structured chat prompts       |
| `StrOutputParser`    | LangChain                    | Parses model output into a string     |
| `ChatOllama`         | LangChain Ollama integration | Connects the chain to an Ollama model |

These names already have implementations. We import and use them rather than defining them ourselves.

### C. Custom functions and variables

These are defined by the developer.

| Name                           | Type                            | Purpose                        |
| ------------------------------ | ------------------------------- | ------------------------------ |
| `DATASET_NAME`                 | Variable/constant by convention | Stores the dataset name        |
| `create_or_get_dataset()`      | Custom function                 | Creates or retrieves a dataset |
| `evaluate_contains_keywords()` | Custom function                 | Checks selected keywords       |
| `evaluate_conciseness()`       | Custom function                 | Checks answer length           |
| `target_qa_chain()`            | Custom function                 | Generates the model's answer   |
| `main()`                       | Custom function                 | Coordinates the program        |
| `prediction`                   | Variable                        | Stores the generated answer    |
| `expected`                     | Variable                        | Stores the reference answer    |
| `word_count`                   | Variable                        | Stores the number of words     |
| `score`                        | Variable                        | Stores an evaluation score     |

A function name is not automatically predefined just because it looks important.

For example:

```python
def main():
    print("Hello")
```

Here, `def` is a Python keyword, `main` is a custom function name, and `print()` is a Python built-in function.

---

## 5. Understanding Methods

A method is a function associated with an object or class.

Our code contains methods from both Python and external libraries.

| Expression                 | Source             | Purpose                        |
| -------------------------- | ------------------ | ------------------------------ |
| `prediction.lower()`       | Python string      | Converts text to lowercase     |
| `expected.split()`         | Python string      | Splits text into parts         |
| `run.outputs.get()`        | Python dictionary  | Retrieves a value using a key  |
| `chain.invoke()`           | LangChain runnable | Executes the pipeline          |
| `client.has_dataset()`     | LangSmith client   | Checks for an existing dataset |
| `client.create_dataset()`  | LangSmith client   | Creates a dataset              |
| `client.read_dataset()`    | LangSmith client   | Retrieves a dataset            |
| `client.create_examples()` | LangSmith client   | Adds examples to a dataset     |

For example:

```python
prediction.lower()
```

Here, `.lower()` is a Python string method.

However:

```python
client.create_dataset(...)
```

Here, `.create_dataset()` is a method provided by the LangSmith client.

Both are methods, but they come from different sources.

---

## 6. Custom Evaluator A: Keyword Verification

Our first evaluator is:

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

### What does it do?

It compares the prediction with the reference answer using a basic keyword check.

### Important variables

* `run`: The recorded execution result supplied to the evaluator.
* `example`: The original dataset example.
* `prediction`: The model's generated answer.
* `expected`: The reference answer.
* `score`: The resulting evaluation score.

### Understanding the scoring logic

```python
expected.split()[:3]
```

This splits the reference answer into words and selects the first three.

For example:

```text
A platform for building, testing, evaluating...
```

The selected words are:

```python
["A", "platform", "for"]
```

The evaluator checks whether any of these words appears in the prediction, ignoring capitalization.

```python
score = 1.0 if any(...) else 0.0
```

A matching word produces a score of `1.0`; otherwise, the score is `0.0`.

### Important limitation

This evaluator does not reliably measure answer correctness.

For example, an incorrect response containing the word "A" could receive a perfect score.

For a better benchmark, we should select meaningful keywords and verify whether the response contains the required concepts. We could also use an LLM-based evaluator or semantic similarity, depending on the evaluation requirements.

---

## 7. Custom Evaluator B: Conciseness

Our second evaluator is:

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

### What does it do?

It checks whether the answer contains 150 words or fewer.

### How does it work?

First, it counts the words:

```python
word_count = len(prediction.split())
```

Then it checks the limit:

```python
is_concise = word_count <= 150
```

Finally, it assigns a score:

```python
score=1.0 if is_concise else 0.5
```

| Condition           | Score |
| ------------------- | ----: |
| 150 words or fewer  |   1.0 |
| More than 150 words |   0.5 |

This evaluator measures length, not correctness.

A short but incorrect answer could still receive a perfect conciseness score.

Also, the system prompt asks the model to respond in one or two sentences, while this evaluator permits up to 150 words. These are different levels of strictness.

---

## 8. Understanding EvaluationResult

Both evaluators return an `EvaluationResult`.

For example:

```python
return EvaluationResult(
    key="conciseness",
    score=1.0,
    comment=f"Word count: {word_count} words."
)
```

This records three pieces of information:

| Field     | Meaning               |
| --------- | --------------------- |
| `key`     | Identifies the metric |
| `score`   | Stores the score      |
| `comment` | Explains the result   |

For example:

```text
Metric: conciseness
Score: 1.0
Comment: Word count: 12 words.
```

`EvaluationResult` is provided by LangSmith. Our evaluator functions decide what score and comment to return.

---

## 9. Understanding the evaluate() Function

This is the central part of the lesson.

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

### What does `evaluate()` do?

It runs the target application against the evaluation dataset, executes the configured evaluators, and records experiment results.

### Understanding its arguments

**Target function**

```python
target_qa_chain
```

The application being evaluated.

**Dataset**

```python
data=DATASET_NAME
```

Identifies the dataset containing the test questions.

**Evaluators**

```python
evaluators=[
    evaluate_contains_keywords,
    evaluate_conciseness
]
```

Specifies which evaluation functions to execute.

**Experiment prefix**

```python
experiment_prefix="qwen3-8b-qa-experiment"
```

Provides a recognizable prefix for the experiment name.

**Metadata**

```python
metadata={
    "model": "qwen3:8b",
    "infrastructure": "local_ollama",
    "eval_version": "v1.0"
}
```

Records information about the model and configuration used in the experiment.

This is useful when comparing different models or configurations.

### Example

Our dataset contains three questions, and we use two evaluators.

That gives us six individual evaluator results when all three examples complete successfully:

Three examples × Two evaluators = Six evaluation results.

The two metrics are recorded separately. They are not automatically combined into a single accuracy score.

---

## 10. What Are Runs in LangSmith?

This was one of our main discussion points.

**A Run is a standard LangSmith concept, not a custom function name.**

A run represents a recorded execution of a component or operation.

Depending on how tracing is configured, you may see executions associated with:

* The overall chain.
* The model invocation.
* Individual traced functions.
* Other components of the application.

### Standard concepts vs. custom names

| Name                         | Classification                        |
| ---------------------------- | ------------------------------------- |
| Run                          | Standard LangSmith concept            |
| Dataset                      | Standard LangSmith concept            |
| Experiment                   | Standard LangSmith evaluation concept |
| `target_qa_chain`            | Custom function                       |
| `evaluate_contains_keywords` | Custom function                       |
| `evaluate_conciseness`       | Custom function                       |
| `qwen3-8b-qa-experiment`     | Custom experiment prefix              |
| `ChatOllama`                 | Library-provided class                |
| `qwen3:8b`                   | Ollama model identifier               |

### Run vs. Experiment

These are related, but they are not interchangeable.

**Run:** Represents an individual recorded execution.

**Experiment:** Represents an evaluation of your application against a dataset, allowing you to inspect predictions and evaluation results.

Your `evaluate()` call coordinates the evaluation workflow and records experiment results.

Not every Python function call automatically becomes a separate visible run. That depends on the tracing and instrumentation configuration.

---

## 11. Local Ollama vs. LangSmith

Another important concept is understanding where each operation happens.

| Operation                            | Location                              |
| ------------------------------------ | ------------------------------------- |
| Running Qwen3:8b inference           | Local Ollama environment              |
| Building the prompt                  | Python/LangChain application          |
| Executing custom evaluator functions | Python process running the evaluation |
| Creating and managing the dataset    | LangSmith service                     |
| Recording experiment results         | LangSmith service                     |
| Inspecting results                   | LangSmith interface                   |

The model runs locally, while LangSmith provides remote dataset management and experiment tracking.

Consequently, evaluation data and predictions may be uploaded to LangSmith. Avoid sending confidential information without checking your organization's policies and the applicable data-handling requirements.

---

## 12. Understanding main()

The `main()` function coordinates the entire program.

Its workflow is:

1. Retrieve the LangSmith API key from the environment.
2. Stop if the key is missing.
3. Initialize the LangSmith client.
4. Create or retrieve the dataset.
5. Run the evaluation.
6. Print the completion message.

The relevant code is:

```python
def main():
    api_key = os.getenv("LANGCHAIN_API_KEY")

    if not api_key:
        print("[ERROR] LANGCHAIN_API_KEY is required to run LangSmith dataset evaluations.")
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
        experiment_prefix="qwen3-8b-qa-experiment"
    )
```

The actual script includes additional metadata and status messages.

### What is the purpose of this line?

```python
if __name__ == "__main__":
    main()
```

It calls `main()` when the file is executed directly.

This is a standard Python pattern.

The `__name__` variable is provided by Python, and `"__main__"` is the conventional string used for direct execution.

---

## 13. The Entire Workflow

Here is the complete conceptual flow.

```text
          LANGSMITH DATASET
        Questions + Reference Answers
                    |
                    v
             evaluate()
                    |
                    v
           target_qa_chain()
                    |
                    v
            Prompt Template
                    |
                    v
              ChatOllama
               Qwen3:8b
                    |
                    v
             Generated Answer
                    |
              +-----+-----+
              |           |
              v           v
        Keyword Check   Length Check
              |           |
              v           v
          Keyword Score  Conciseness Score
              |           |
              +-----+-----+
                    |
                    v
          LANGSMITH EXPERIMENT
           Predictions + Scores
                    |
                    v
           Inspect in LangSmith
```

---

## 14. Five Interview Questions to Revise

### Q1. What is a LangSmith dataset?

A collection of evaluation examples containing inputs and, optionally, reference outputs used to test an LLM application.

### Q2. What is a target function?

The application or callable being evaluated. In our script, `target_qa_chain()` generates answers using a local Qwen model through Ollama.

### Q3. What is a custom evaluator?

A function that evaluates an application's output against a defined criterion, such as keyword matching or response length.

### Q4. What does evaluate() do?

It executes the target application against a dataset, runs the specified evaluators, and records the evaluation results as an experiment.

### Q5. What is the difference between a Run and an Experiment?

A run represents a recorded execution. An experiment represents the evaluation of an application against a dataset, with results that can be inspected and compared.

---

## 15. Final Revision: What You Should Remember

There are six key ideas in this lesson:

1. **Dataset:** Contains the questions and reference answers.
2. **Target function:** Generates predictions using the model.
3. **Evaluator:** Checks a particular quality criterion.
4. **EvaluationResult:** Represents the evaluator's score and comment.
5. **evaluate():** Coordinates the evaluation process.
6. **Experiment:** Organizes the evaluation results for inspection and comparison in LangSmith.

And remember our earlier discussion about naming:

* Python keywords and built-ins come from Python.
* Classes and functions such as `Client`, `evaluate()`, and `ChatOllama` come from libraries.
* Functions such as `main()` and `target_qa_chain()` are custom functions defined in our code.
* Names such as `DATASET_NAME`, `prediction`, and `score` are variables created by the developer.
* Runs and experiments are standard LangSmith concepts, although custom names can be associated with them.

**The central takeaway:** We are learning how to create a repeatable evaluation process that tests an AI application's answers against a dataset and records measurable results, rather than relying entirely on manual inspection.
