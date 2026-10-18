# Lesson 02: LangGraph Workflow + LangSmith Tracing

## 1. First, understand what we're building

Imagine you're building an AI customer support system.

A customer sends one of these messages:

* "I'm getting an error when I submit the form."
* "I need a copy of my invoice."
* "What are your business hours?"

Your system must identify the type of query and send it to the correct department.

Here's what your program does:

![Image](https://images.openai.com/static-rsc-4/zgoNbNa8rxEbC1C0kQ9YsBrzGeBAExbOoTJkCL1EMf9CJ99XWBamSD3_3IaQe6ickNOKyDtzPpSlS77mM44abrtNZaK2Ok-9_09LUDWyDcFGtsJWw8w6nrcUNjh_U2o1Ruv0Jp47iZSYrQOwwxeXposTDU1Y94WFkyTLvLP55TBLYrKBCaeqS93XexPfK5Ye?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/WN4p0-dmHjkBK2j0-6TAyw1vhqInn5_3ayHdrlUUGm-f-EsvSw0eJJMpzasXaSnQU8nUM1B_-10MkYDNwGWYd2QXNDvl3pbpyFIT9zs7BHrbsHsl8_XBhxA9s4acwAg1NYZkE3Buyv5-Slc0gIfa632RKOpCY9AWBYkjDc9ucYjY2XMb0Clui_qpNwZlV4Rk?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/wiSKyxOKbKDyDmiaCXv7oQhuOgiyn6VYyyCSac5Wb-tPpB3YHRG76IX5LeNT_j0n9AZNEB49mBReGI3pZUlIRAwyau3zhWX-ilUskPZpPwCrWt4NygkKWgPqpSSzRDnL3dnq3k0uSybyNQBwquPv3v5cPmcn5WI88jPsye-fCHUAWnkt53NJXGQMIfzCEdKX?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/AMni4vgAj02vBCH7gShRdwFcdZA7oK3hSsJOYHH-1QcO1FG8Elyoghzb_2BsnO7mSMg6-9Bv2ubzluJIH24oV0TRIULZXs_kOg1vEUUrWf1SEMidp0LmrqX-v9ClCom9H3lEY_Y28V0zNyFgh5MRU1c15IBeCy7bLPxjR1g-OQ3e9ypyLdV93Se5JaZ0mbFq?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/VbZP3pYmLavQUNfgtKSE3c8xDoa1K33LoWDiZbSL3dPSg-pELufBFRHj1ZCEPK9VhtFefQNbD2f8GzIj6eXB1sBmkmsuui2gy0fC7osBXMkJoXz66HqAhHrldMTTTIlHPGZvJVj4AQ3-GJXzCR43En743ostplQENOvPzu1JEVcD1Fhy_iDFqo12EyPl-tdu?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/fw1owhEMTHggZyE_EQQIFW_1jtdsfthYKpHPCrVQtOW8Px4-Cy_izpJGfLEheEZUUI9jEz1zmoOwOYIc6dcPDa2MsNRYHejsvgk0OXKV0xuw3-GrJZHFheZ8h1auml7FUG_5XtWibCXekYLEEkMH-uw6s1a3FI33yUrGvW6VTYB6u_hVVJetONh4J7JxHwbY?purpose=fullsize)

![Image](https://images.openai.com/static-rsc-4/fbOLRNxuunag79kLAoUdaTc9pYcjbDKGiJfgvk108uaslDli09htvbXWV_30aVkr3rqByIuYQtyKovb57nErhzp5znsuT2z2_Zw_VGPFEipA7Awmg5PBhhrGB0VkkAW0Xos5jahn2pCy5x_G3vQ_3Ap9Q6ohiknTt65-gYdkVPWhcqQ_Tu40QRLkQ-vV07RX?purpose=fullsize)

The workflow looks like this:

```text
                  START
                    |
                    v
            +----------------+
            |   CLASSIFIER   |
            |                |
            | Analyze query  |
            +----------------+
                    |
             Check category
                    |
          +---------+---------+
          |         |         |
          v         v         v
     TECHNICAL   BILLING   GENERAL
      SUPPORT
          |         |         |
          v         v         v
         END       END       END
```

For example:

**Input:**

```text
I got a NullPointer exception when clicking submit.
```

**Processing:**

```text
1. Receive the query.
2. Analyze the query.
3. Detect the word "exception".
4. Assign category = technical_support.
5. Route the query to the technical support node.
6. Generate the response.
7. Finish execution.
```

LangGraph manages the workflow, while LangSmith helps you inspect its execution.

Let's understand how.

---

# 2. Understand the three core concepts

Before examining the code, remember these three definitions.

| Concept   | What it does                                  | Analogy                      |
| --------- | --------------------------------------------- | ---------------------------- |
| LangGraph | Controls the workflow                         | A company's workflow manager |
| State     | Stores information as the workflow progresses | A shared case file           |
| LangSmith | Records and displays execution traces         | A monitoring dashboard       |

### LangGraph

LangGraph allows you to define a workflow using nodes and edges.

* **Nodes:** Functions that perform tasks.
* **Edges:** Connections that determine what happens next.
* **Conditional edges:** Connections that select the next node based on a condition.
* **State:** Information passed between nodes.

### State

State is the information available to your workflow.

For your program, it contains the original query, the detected category, the reasoning, the generated response, and the execution logs.

### LangSmith

LangSmith helps you inspect a workflow's execution, including:

* Which nodes executed.
* The order in which they executed.
* How long operations took.
* Inputs and outputs captured by tracing.
* Errors and failures.
* Nested operations and their relationships.

**Important distinction:** LangGraph executes the workflow. LangSmith observes and records it. LangSmith is not responsible for choosing the next node in your graph.

Now let's examine your code.

---

# 3. Step 1: Why do you import `config` first?

Your code begins with:

```python
import config

import sys
import os
```

The comment says:

```python
# CRITICAL: Import config FIRST to set os.environ before LangChain initializes!
```

Let's understand why.

## What is `config.py`?

In your project, `config.py` presumably configures environment variables for LangSmith tracing.

For example, it might contain:

```python
import os

os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_API_KEY"] = "your-api-key"
os.environ["LANGSMITH_PROJECT"] = "langgraph-demo"
```

This is an illustrative example. Your actual configuration might use different settings.

### What is `os.environ`?

It's Python's interface for accessing environment variables.

Think of environment variables as configuration settings available to your running program.

For example:

```python
import os

os.environ["APP_MODE"] = "development"

print(os.getenv("APP_MODE"))
```

Output:

```text
development
```

### Why import `config` first?

Python executes the statements in a module when that module is first imported.

Therefore:

```python
import config
```

executes your configuration code before Python proceeds to the subsequent imports.

This ensures that the environment variables are set early enough for libraries that read them during initialization.

The exact initialization behavior depends on the libraries involved, but loading configuration early is a good practice.

**One important point:** Your script imports `langgraph` and `langsmith` directly. It doesn't explicitly import `langchain` here. The comment's purpose is to ensure that tracing configuration is ready before the relevant libraries initialize.

### How does tracing get enabled?

Conceptually:

```text
Your Python program starts
          |
          v
    import config
          |
          v
 Environment variables set
          |
          v
 LangGraph and LangSmith imported
          |
          v
 Graph executes
          |
          v
 Execution data is traced
```

For LangSmith tracing to work, you generally need the appropriate tracing setting enabled, a valid API key, and the intended project configured.

The API key should be kept secret rather than hardcoded into shared source code.

---

# 4. Step 2: Understand your imports

Your code contains:

```python
import sys
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langsmith import traceable, Client
```

Let's break these down.

| Import       | Purpose                                                           |
| ------------ | ----------------------------------------------------------------- |
| `sys`        | Provides access to Python interpreter and runtime information     |
| `os`         | Accesses environment variables and operating-system functionality |
| `TypedDict`  | Defines the expected structure of a dictionary                    |
| `List`       | Provides a type annotation for a list                             |
| `StateGraph` | Creates a graph-based workflow                                    |
| `START`      | Represents the graph's entry point                                |
| `END`        | Represents the graph's termination point                          |
| `traceable`  | Adds tracing to selected functions                                |
| `Client`     | Provides access to LangSmith's client functionality               |

Two details are worth noticing.

First, `sys` is imported but not used in the code you've shared. You can remove it without changing the current program's behavior.

Second, `TypedDict` and `List` are used for type annotations. They describe the expected data structure; they don't automatically validate every value at runtime.

---

# 5. Step 3: Understand the State Schema

This is one of the most important concepts in LangGraph.

Your code:

```python
class AgentState(TypedDict):
    query: str
    category: str
    reasoning: str
    response: str
    logs: List[str]
```

Let's understand it from zero.

## What is a state?

Imagine a customer support ticket.

Initially, you have the customer's message:

```python
{
    "query": "I got an exception when clicking submit",
    "logs": []
}
```

As your workflow progresses, it determines the category and produces a response.

The resulting state might look like this:

```python
{
    "query": "I got an exception when clicking submit",
    "category": "technical_support",
    "reasoning": "Detected error keywords in query.",
    "response": "[Tech Support Desk]: We have logged your technical issue...",
    "logs": [
        "Query analyzed by classifier node.",
        "Routed to Tech Support."
    ]
}
```

Think of the state as a **shared case file that travels through the workflow**.

Each node can read relevant information from it and return updates.

## What does `TypedDict` do?

This:

```python
class AgentState(TypedDict):
    query: str
    category: str
    reasoning: str
    response: str
    logs: List[str]
```

describes the expected dictionary structure.

Here's what each field means:

| Field       | Type        | Purpose                                |
| ----------- | ----------- | -------------------------------------- |
| `query`     | `str`       | Original customer message              |
| `category`  | `str`       | Classification result                  |
| `reasoning` | `str`       | Explanation produced by the classifier |
| `response`  | `str`       | Final support response                 |
| `logs`      | `List[str]` | Messages recording workflow actions    |

For example:

```python
"query": "I need my invoice"
```

The value is a string.

```python
"logs": ["Query received"]
```

The value is a list of strings.

### A subtle but important detail

Your initial input is:

```python
{"query": q, "logs": []}
```

It doesn't contain `category`, `reasoning`, or `response`.

That's fine for this workflow because the classifier produces the first two fields, and the selected handler produces the response.

`AgentState` describes the state schema, but it doesn't mean that every field must already exist in the initial input dictionary. In particular, your code doesn't define `total=False` or mark fields individually as optional.

### Why is state important?

Without state, every node would need another way to receive information from previous steps.

With state:

```text
Input
  |
  v
Classifier
  |
  | Returns category and reasoning
  v
Updated State
  |
  v
Selected Handler
  |
  | Returns response and logs
  v
Final State
```

**Remember this:** Nodes perform work; state carries information between them.

---

# 6. Step 4: Understand what a node is

In LangGraph, a node is a function or runnable that performs a specific task.

Your program defines four nodes:

| Node                | Responsibility            |
| ------------------- | ------------------------- |
| `route_query_node`  | Classifies the query      |
| `tech_support_node` | Handles technical issues  |
| `billing_node`      | Handles billing questions |
| `general_node`      | Handles general questions |

Let's examine the classifier first.

## The classifier node

Your code:

```python
@traceable(name="Analyze & Route Node", run_type="chain")
def route_query_node(state: AgentState) -> dict:
    query = state["query"].lower()
    logs = state.get("logs", [])
    logs.append("Query analyzed by classifier node.")

    if any(k in query for k in ["bug", "error", "exception", "failed"]):
        category = "technical_support"
        reasoning = "Detected error keywords in query."
    elif any(k in query for k in ["price", "billing", "invoice", "refund"]):
        category = "billing"
        reasoning = "Detected financial/billing keywords."
    else:
        category = "general"
        reasoning = "No specialized keywords matched, defaulting to general."

    return {
        "category": category,
        "reasoning": reasoning,
        "logs": logs
    }
```

We'll break it into four parts.

### Part A: The decorator

```python
@traceable(name="Analyze & Route Node", run_type="chain")
```

This decorator comes from LangSmith.

A decorator lets you wrap a function with additional behavior.

In this case, `@traceable` instruments the function so its execution can appear in LangSmith traces when tracing is configured correctly.

The name:

```python
name="Analyze & Route Node"
```

provides a readable name for the traced operation.

The run type:

```python
run_type="chain"
```

categorizes the operation as a chain-type run in LangSmith.

This is a tracing classification. It doesn't turn your function into a language model or a chain.

**What you'll see:** The traced operation can appear under the name `Analyze & Route Node` in your LangSmith trace.

### Part B: Read the query

```python
query = state["query"].lower()
```

Suppose the original query is:

```text
I got a NullPointer exception when clicking submit
```

The `.lower()` method converts it to:

```text
i got a nullpointer exception when clicking submit
```

Why?

So your keyword matching doesn't depend on capitalization.

For example:

```python
"ERROR" in "error occurred"
```

returns `False`, but converting both strings to lowercase avoids this particular problem.

Next:

```python
logs = state.get("logs", [])
```

This retrieves the existing logs.

If the `logs` field is missing, it uses an empty list.

Then:

```python
logs.append("Query analyzed by classifier node.")
```

This adds an entry to the log.

### Part C: Classify the query

Consider:

```python
if any(k in query for k in ["bug", "error", "exception", "failed"]):
```

Let's simplify the inner expression:

```python
any(k in query for k in ["bug", "error", "exception", "failed"])
```

It checks whether **at least one** of the specified keywords appears in the query.

For example:

```python
query = "I got a nullpointer exception"
```

The checks are conceptually:

```text
"bug"       in query? False
"error"     in query? False
"exception" in query? True
"failed"    in query? False
```

Since one keyword matches, `any()` returns `True`.

The classifier assigns:

```python
category = "technical_support"
reasoning = "Detected error keywords in query."
```

The billing condition works similarly:

```python
elif any(k in query for k in ["price", "billing", "invoice", "refund"]):
```

If neither condition matches, the query goes to the general category.

**Important limitation:** This is a rule-based classifier, not an LLM-based classifier. It detects keywords rather than understanding the query semantically.

For example, the word `exception` routes to technical support, but a query such as "I want to make an exception to my billing policy" might also be routed incorrectly.

That's a useful limitation to remember when you later build more sophisticated agents.

### Part D: Return updates to the state

```python
return {
    "category": category,
    "reasoning": reasoning,
    "logs": logs
}
```

This is critical.

The node returns a dictionary containing the fields it wants to update.

For example:

```python
{
    "category": "technical_support",
    "reasoning": "Detected error keywords in query.",
    "logs": ["Query analyzed by classifier node."]
}
```

LangGraph applies these updates to the graph's state.

The original query remains available, and the new fields become available to subsequent nodes.

**A node generally doesn't need to return the entire state. It can return only the fields it updates.**

---

# 7. Step 5: Understand the handler nodes

After classification, the graph selects one handler.

Let's examine the technical support handler.

```python
@traceable(name="Tech Support Handler Node", run_type="tool")
def tech_support_node(state: AgentState) -> dict:
    return {
        "response": f"[Tech Support Desk]: We have logged your technical issue ('{state['query']}'). Check system logs for stack traces.",
        "logs": state.get("logs", []) + ["Routed to Tech Support."]
    }
```

Notice that the handler doesn't classify the query again.

It assumes the classifier has already done that.

It reads:

```python
state["query"]
```

and creates a response:

```python
"response": f"...{state['query']}..."
```

It also appends a log entry:

```python
state.get("logs", []) + ["Routed to Tech Support."]
```

Here, `+` creates a new list containing the previous log entries and the new entry.

The handler returns only:

```python
{
    "response": "...",
    "logs": [...]
}
```

LangGraph combines these updates with the existing state.

### Why use separate handlers?

Because each node has a distinct responsibility.

The technical support handler doesn't need to know how to format billing responses.

The billing handler doesn't need to know how to answer general questions.

This separation makes the workflow easier to maintain, test, and extend.

Your other two handlers work the same way:

```python
def billing_node(state: AgentState) -> dict:
    ...
```

```python
def general_node(state: AgentState) -> dict:
    ...
```

They differ in the response they generate and the log message they append.

### What does `run_type="tool"` mean here?

Your handler functions are annotated with:

```python
@traceable(name="Tech Support Handler Node", run_type="tool")
```

This labels their traced runs as tool-type operations.

However, these functions aren't actually calling an external support system. They're ordinary Python functions that generate strings.

The tracing label doesn't itself connect them to a ticketing platform or make them external tools.

---

# 8. Step 6: Understand the conditional router

This is the decision-making point of your graph.

Your code:

```python
def router_conditional_edge(state: AgentState) -> str:
    """Conditional router function determining target node based on category."""
    return state["category"]
```

This function reads the classification result.

Suppose the state contains:

```python
{
    "query": "I got an exception",
    "category": "technical_support"
}
```

The router returns:

```python
"technical_support"
```

That's it.

It doesn't execute the technical support handler itself. It tells LangGraph which route to take.

Your graph defines the mapping separately:

```python
builder.add_conditional_edges(
    "classifier",
    router_conditional_edge,
    {
        "technical_support": "technical_support",
        "billing": "billing",
        "general": "general"
    }
)
```

Let's understand the three arguments.

| Argument                  | Meaning                                          |
| ------------------------- | ------------------------------------------------ |
| `"classifier"`            | The node after which the routing decision occurs |
| `router_conditional_edge` | The function that determines the route           |
| The dictionary            | Maps returned values to destination nodes        |

For example:

```text
Router returns "billing"
          |
          v
Mapping finds "billing"
          |
          v
Billing handler executes
```

### Why does the mapping matter?

Your router returns a string.

LangGraph uses that string to select a destination through the mapping.

If the router returns a value that isn't mapped, the graph won't have a valid destination for that decision.

This is why the values returned by the router and the keys in the mapping must match.

### A distinction worth remembering

The router is not a node in this graph.

It's a routing function associated with a conditional edge.

The classifier is a node. The routing function examines the resulting state and determines which node runs next.

---

# 9. Step 7: Build the graph with `StateGraph`

Now we reach the part that connects everything.

Your function:

```python
def build_agent_graph():
    builder = StateGraph(AgentState)
```

This creates a graph builder using `AgentState` as the graph's state schema.

Think of `builder` as the blueprint you're assembling.

At this point, you've created the builder, but you haven't yet connected all the nodes.

## A. Add the nodes

```python
builder.add_node("classifier", route_query_node)
builder.add_node("technical_support", tech_support_node)
builder.add_node("billing", billing_node)
builder.add_node("general", general_node)
```

The pattern is:

```python
builder.add_node("node_name", function)
```

For example:

```python
builder.add_node("classifier", route_query_node)
```

This tells LangGraph that the node named `classifier` should execute `route_query_node`.

The string is the graph's node identifier. The function is the work performed by that node.

## B. Connect the starting point

```python
builder.add_edge(START, "classifier")
```

This defines the first transition:

```text
START → CLASSIFIER
```

When you invoke the graph, execution begins at `START` and proceeds to the classifier.

## C. Add conditional edges

```python
builder.add_conditional_edges(
    "classifier",
    router_conditional_edge,
    {
        "technical_support": "technical_support",
        "billing": "billing",
        "general": "general"
    }
)
```

This defines the possible next steps after the classifier.

The selected handler depends on the routing function's return value.

## D. Connect each handler to the end

```python
builder.add_edge("technical_support", END)
builder.add_edge("billing", END)
builder.add_edge("general", END)
```

These edges terminate the workflow after the chosen handler finishes.

The graph doesn't execute all three handlers. It executes the selected handler and then reaches `END`.

## E. Compile the graph

Finally:

```python
return builder.compile()
```

Compilation turns the graph definition into an executable graph.

The overall process is:

```text
1. Create graph builder
          |
          v
2. Register nodes
          |
          v
3. Connect edges
          |
          v
4. Configure conditional routing
          |
          v
5. Compile the graph
          |
          v
6. Obtain executable graph
```

The compiled graph is then ready to be invoked.

### One important distinction

**Building a graph defines the workflow. Invoking a graph executes it.**

Compilation doesn't mean the customer support queries have already been processed.

---

# 10. Step 8: Understand `graph.invoke()`

Your `main()` function builds the graph:

```python
graph = build_agent_graph()
```

Then it defines three test queries:

```python
test_queries = [
    "I got a NullPointer exception when clicking submit",
    "Can I get a copy of my monthly invoice?",
    "What are your business hours?"
]
```

Next, it iterates over them:

```python
for idx, q in enumerate(test_queries, 1):
```

The first query gets `idx = 1`, the second gets `idx = 2`, and the third gets `idx = 3`.

Now look at the invocation:

```python
final_state = graph.invoke(
    {"query": q, "logs": []},
    config={
        "run_name": f"LangGraph Support Ticket Routing #{idx}",
        "tags": ["langgraph", "support-bot", "routing-demo"],
        "metadata": {"test_case_id": f"TC-00{idx}"}
    }
)
```

This is where the graph executes.

Let's examine both arguments.

## A. The initial state

```python
{"query": q, "logs": []}
```

This supplies the input to the graph.

For the first query, it is equivalent to:

```python
{
    "query": "I got a NullPointer exception when clicking submit",
    "logs": []
}
```

The classifier receives this state.

## B. The configuration

```python
config={
    "run_name": f"LangGraph Support Ticket Routing #{idx}",
    "tags": ["langgraph", "support-bot", "routing-demo"],
    "metadata": {"test_case_id": f"TC-00{idx}"}
}
```

This provides run configuration, including useful tracing information.

| Setting    | Purpose                                         |
| ---------- | ----------------------------------------------- |
| `run_name` | Gives the invocation a readable name            |
| `tags`     | Adds labels that help organize and filter runs  |
| `metadata` | Adds structured information about the test case |

For the first query, the run name becomes:

```text
LangGraph Support Ticket Routing #1
```

And the metadata contains:

```python
{"test_case_id": "TC-001"}
```

This is useful when you want to distinguish individual test executions in LangSmith.

These settings describe the run; they don't change the classifier's keyword-matching rules.

### What does `graph.invoke()` return?

It returns the final state produced by the completed graph execution.

Your code then prints:

```python
print(f"Category : {final_state['category']}")
print(f"Reasoning: {final_state['reasoning']}")
print(f"Response : {final_state['response']}")
print(f"Logs     : {final_state['logs']}\n")
```

Let's predict the results.

| Query                 | Category            | Handler           |
| --------------------- | ------------------- | ----------------- |
| NullPointer exception | `technical_support` | Technical support |
| Monthly invoice       | `billing`           | Billing           |
| Business hours        | `general`           | General           |

The first query matches `exception`.

The second matches `invoice`.

The third matches none of the specialized keywords, so it falls back to `general`.

---

# 11. Step 9: Now, how exactly does LangSmith work?

This is the part you especially want to understand.

Think of LangSmith as an observability platform for your LLM applications and workflows.

It helps you answer questions like:

* Did my graph execute?
* Which node ran first?
* Which handler was selected?
* What input did a node receive?
* What output did it produce?
* How long did each operation take?
* Where did an error occur?

Let's separate the underlying concepts.

## A. Tracing

A **trace** represents an execution of an operation or a larger workflow, along with its recorded child operations.

When you run:

```python
final_state = graph.invoke(...)
```

LangGraph executes the graph.

With tracing enabled and configured correctly, LangSmith can record information about that invocation.

Conceptually:

```text
LangGraph invocation
        |
        v
   Classifier node
        |
        v
  Conditional routing
        |
        v
   Selected handler
        |
        v
    Final result
```

LangSmith records trace data that helps you inspect this execution.

The exact trace tree depends on the tracing configuration, library integration, and which operations are instrumented.

## B. Runs

In LangSmith, a run represents an individual operation within a trace.

For example, a trace may contain a run for the overall graph invocation and child runs for individual nodes or traced functions.

Your decorator:

```python
@traceable(name="Analyze & Route Node", run_type="chain")
```

instruments the classifier function.

Likewise:

```python
@traceable(name="Tech Support Handler Node", run_type="tool")
```

instruments the technical support handler.

This allows you to inspect those operations as part of the recorded execution.

A run can contain information such as:

* Its name and type.
* Inputs and outputs.
* Start and end times.
* Duration.
* Error information, if one occurred.
* Relationships to parent and child runs.

Not every field is necessarily populated in every run.

## C. The trace hierarchy

A simplified example of the first query might look like this:

```text
LangGraph Support Ticket Routing #1
|
+-- Analyze & Route Node
|
+-- Tech Support Handler Node
```

The classifier and handler are the two explicitly decorated functions in this execution path.

The graph invocation may also appear as an enclosing run, depending on the LangGraph tracing integration and configuration.

Notice what is missing:

```text
Billing Handler Node
General Handler Node
```

Those nodes weren't executed for this query.

That's the key insight: **the trace lets you inspect what actually happened, rather than just what your graph could do.**

## D. How does the data get to LangSmith?

The conceptual flow is:

```text
Your Python application
          |
          v
  LangGraph executes
          |
          v
Tracing captures execution
          |
          v
 LangSmith client/integration
          |
          v
  LangSmith backend
          |
          v
   LangSmith dashboard
```

The exact transport and batching behavior depend on the SDK and configuration.

Your `config.py` likely enables tracing and supplies the credentials and project configuration needed for the data to reach the appropriate LangSmith project.

The dashboard then lets you inspect the recorded runs.

### What LangSmith does NOT do

LangSmith doesn't need to classify your query itself.

It doesn't choose the billing or technical support node.

It doesn't execute your business logic on behalf of LangGraph.

It records execution information so you can understand, debug, and evaluate your application.

---

# 12. Step 10: What is the purpose of `Client()` and `flush()`?

Your script ends with:

```python
print("[FLUSHING TRACES] Synchronizing trace logs with LangSmith...")

try:
    client = Client()
    client.flush()
except Exception as e:
    print(f"[NOTE] Trace flush warning: {e}")
```

The intention is to synchronize any pending trace data before the program finishes.

However, **there's a potential issue with this particular code**.

In the standard LangSmith Python SDK, you shouldn't assume that `Client()` provides a `flush()` method. If the installed SDK version doesn't provide it, this code will raise an `AttributeError`.

Because the exception is caught, the program will print a warning rather than crash.

The tracing integration may already be managing its own background processing. Whether an explicit flush is necessary depends on the SDK and version you're using.

### How should you handle this?

Check the API supported by your installed LangSmith version.

You can inspect the client in your local environment:

```python
from langsmith import Client

client = Client()

print(type(client))
print(hasattr(client, "flush"))
```

If the final line prints:

```text
False
```

your client doesn't expose that method.

Don't add a manual flush call unless the installed SDK actually supports the method you're calling.

For reliable tracing, follow the documented behavior for the specific LangSmith SDK version in your environment.

---

# 13. Step 11: Understand the project name

Your code:

```python
project_name = os.getenv("LANGCHAIN_PROJECT", "default")
```

This reads the `LANGCHAIN_PROJECT` environment variable.

If the variable isn't set, it uses:

```text
default
```

Then:

```python
print(
    f"[SUCCESS] All queries processed. "
    f"View traces in project '{project_name}' on LangSmith Dashboard."
)
```

prints a message telling you which project name to look for.

There is one detail to keep in mind.

Your code reads `LANGCHAIN_PROJECT`, but the project setting may also depend on the LangSmith/LangChain environment variables and configuration used by your installed versions.

If the environment variable used for tracing differs from the one used in this final print statement, the displayed project name could be misleading.

The printed message alone doesn't confirm that traces were successfully uploaded.

**To verify that tracing is working, check the actual project and runs in LangSmith.**

---

# 14. Let's follow one query through the entire program

This is where all the concepts come together.

Consider:

```python
q = "Can I get a copy of my monthly invoice?"
```

### Stage 1: Invocation

The program executes:

```python
graph.invoke(
    {"query": q, "logs": []},
    config={...}
)
```

Initial state:

```python
{
    "query": "Can I get a copy of my monthly invoice?",
    "logs": []
}
```

### Stage 2: Classifier

The graph executes:

```python
route_query_node(state)
```

The query is converted to lowercase.

The classifier checks the technical support keywords. None match.

It then checks the billing keywords.

The word `invoice` matches.

The node returns:

```python
{
    "category": "billing",
    "reasoning": "Detected financial/billing keywords.",
    "logs": ["Query analyzed by classifier node."]
}
```

LangGraph applies these updates to the state.

### Stage 3: Conditional routing

The routing function returns:

```python
"billing"
```

LangGraph uses the conditional mapping and selects the billing handler.

### Stage 4: Billing handler

The graph executes:

```python
billing_node(state)
```

The handler generates a response and appends another log entry.

Its updates look like this:

```python
{
    "response": "[Billing Desk]: Please visit account billing settings to review charges for: 'Can I get a copy of my monthly invoice?'.",
    "logs": [
        "Query analyzed by classifier node.",
        "Routed to Billing."
    ]
}
```

### Stage 5: End

The graph follows:

```text
billing → END
```

It returns the final state.

### Stage 6: LangSmith inspection

If tracing is configured correctly, you can inspect the recorded execution in the relevant LangSmith project.

You should be able to investigate the classifier and billing handler operations and see the execution path.

The conceptual result is:

```text
Trace: LangGraph Support Ticket Routing #2

  1. Analyze & Route Node
     Output category: billing

  2. Billing Handler Node
     Output: Billing response

  Result: Completed
```

This is an illustrative representation of the execution, not a screenshot or a guarantee of the exact dashboard layout.

---

# 15. How to inspect your traces in LangSmith

Let's turn the concepts into practical steps.

### Step 1: Verify your configuration

Check that your `config.py` sets the appropriate tracing environment variables and that your LangSmith API key is valid.

For current configuration guidance, use the official [LangSmith tracing documentation](https://docs.langchain.com/langsmith/trace-with-langchain?utm_source=chatgpt.com).

### Step 2: Run your Python file

Your current command is:

```powershell
R:\LangChain-Book\.venv\Scripts\python.exe 02_langgraph_tracing.py
```

Run it from the directory containing the file, or provide the file's full path.

### Step 3: Check the terminal

You should see three queries processed, each with a category, reasoning, response, and logs.

The expected categories are:

```text
Query #1 → technical_support
Query #2 → billing
Query #3 → general
```

If an exception occurs, inspect the error rather than relying only on the final success message.

### Step 4: Open LangSmith

Open the [LangSmith application](https://smith.langchain.com/?utm_source=chatgpt.com) and navigate to the project configured for your tracing setup.

### Step 5: Find the runs

Look for the invocations named:

```text
LangGraph Support Ticket Routing #1
LangGraph Support Ticket Routing #2
LangGraph Support Ticket Routing #3
```

These names come from your `run_name` configuration.

If they don't appear, check your API key, tracing settings, project name, network access, and any warnings or errors in the terminal.

### Step 6: Open a trace

Inspect the available inputs, outputs, execution hierarchy, and timing information.

For the first query, focus on the classifier and technical support handler.

For the second query, focus on the classifier and billing handler.

For the third query, focus on the classifier and general handler.

This will help you connect the Python code to the recorded execution.

---

# 16. One subtle Python issue in your code

Look at these two lines:

```python
logs = state.get("logs", [])
logs.append("Query analyzed by classifier node.")
```

The first line doesn't necessarily create a new list. If `state["logs"]` already exists, `logs` refers to the same list object.

The second line modifies that list in place.

In your current workflow, this is likely to work as intended, but it can be a source of confusion when debugging state updates.

A cleaner approach is:

```python
logs = state.get("logs", []) + ["Query analyzed by classifier node."]
```

This creates a new list rather than modifying the existing list in place.

The return statement can then remain:

```python
return {
    "category": category,
    "reasoning": reasoning,
    "logs": logs
}
```

This makes the state update easier to reason about.

Also, because `logs` is a list, remember that changing a list in place is different from returning a new list. Understanding that distinction will help when you build more complicated LangGraph workflows.

---

# 17. What you've learned

Let's summarize the entire lesson.

| Concept                   | What you should remember                                              |
| ------------------------- | --------------------------------------------------------------------- |
| `config`                  | Sets up configuration before the relevant libraries initialize        |
| `TypedDict`               | Describes the expected structure of the state                         |
| `AgentState`              | Defines the information available to graph nodes                      |
| Node                      | A function or runnable that performs a task                           |
| `@traceable`              | Instruments a function for LangSmith tracing                          |
| `StateGraph`              | Creates a graph workflow                                              |
| `add_node()`              | Registers a node in the graph                                         |
| `add_edge()`              | Connects two graph points                                             |
| `add_conditional_edges()` | Defines conditional routing                                           |
| `START`                   | Marks the graph's entry point                                         |
| `END`                     | Marks the graph's termination point                                   |
| `compile()`               | Turns the graph definition into an executable graph                   |
| `invoke()`                | Executes the graph for an input                                       |
| `run_name`                | Names a graph invocation for easier identification                    |
| `tags`                    | Labels a run for organization and filtering                           |
| `metadata`                | Attaches additional structured information to a run                   |
| LangSmith                 | Records and displays execution traces for debugging and observability |

## The most important distinction

**LangGraph is the workflow engine. LangSmith is the observability platform.**

Your graph determines what happens. LangSmith helps you understand what happened.

One final point: your current program is a rule-based router, not yet a fully LLM-powered agent. That's actually a great starting point because you can learn graph execution, state management, routing, and tracing without introducing model behavior as another variable.

---

# 🧠 Quick knowledge check

Let's make sure the concepts have clicked before moving on.

Try answering these five questions without looking back.

**Q1.** What is the difference between a node and an edge in LangGraph?

**Q2.** In your code, who decides which handler runs: `router_conditional_edge()` or LangSmith?

**Q3.** What does `graph.invoke()` return?

**Q4.** Why do you use `@traceable` on the classifier and handler functions?

**Q5.** If a query contains the word `"invoice"`, explain the complete path it follows through your graph, from input to `END`.

