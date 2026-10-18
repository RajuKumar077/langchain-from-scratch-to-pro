"""
Lesson 07 - User Feedback Loops & Human Annotation Queues
=========================================================
LangSmith Principle:
  Observability without a feedback loop is just passive logging.
  In production LLM applications:
  1. Users provide implicit or explicit feedback (Thumbs Up/Down, Star Ratings, Corrections).
  2. The application MUST capture the unique `run_id` of the LLM execution and record
     feedback against that exact run via `client.create_feedback()`.
  3. When runs fail or receive negative feedback, automated rules or triage logic can
     route those runs into a LangSmith "Annotation Queue" for human expert review.

Run command:
  r:\\LangChain-Book\\.venv\\Scripts\\python.exe 07_feedback_and_annotation_queues.py
"""

import os
import config
from langsmith import Client
from langchain_core.tracers.context import collect_runs
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama

QUEUE_NAME = "Support Triage - Review Queue"

def run_application_and_capture_id(user_query: str) -> tuple[str, str]:
    """
    Step 1: Execute the chain while capturing the active `run_id` using `collect_runs()`.
    This `run_id` is the primary key linking this execution to downstream user feedback.
    """
    print(f"\n[1. EXECUTION] Processing user query: '{user_query}'...")
    
    llm = ChatOllama(model=os.getenv("OLLAMA_MODEL", "qwen3:8b"), temperature=0.3)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a customer service assistant. Provide a helpful, concise answer."),
        ("user", "{query}")
    ])
    chain = prompt | llm | StrOutputParser()

    # CRITICAL: collect_runs() captures all traced runs in this execution context
    with collect_runs() as cb:
        response = chain.invoke(
            {"query": user_query},
            config={
                "run_name": "Customer Support Query",
                "tags": ["production-traffic", "support-bot"],
                "metadata": {"customer_tier": "enterprise", "channel": "web_chat"}
            }
        )
        # Extract the root run ID
        run_id = str(cb.traced_runs[0].id)

    print(f" -> Generated Response: {response.strip()[:100]}...")
    print(f" -> Captured LangSmith Run ID: {run_id}")
    return run_id, response


def record_user_feedback(client: Client, run_id: str, is_positive: bool, user_comment: str = ""):
    """
    Step 2: Attach user feedback to the run in LangSmith.
    Supports binary thumbs up/down, numeric scores (1-5), and qualitative comments.
    """
    print(f"\n[2. USER FEEDBACK] Attaching feedback to Run ID [{run_id[:8]}...]...")
    
    # Record binary thumbs rating
    client.create_feedback(
        run_id=run_id,
        key="user_thumbs",
        score=1.0 if is_positive else 0.0,
        comment=user_comment,
        source_info={"source": "customer_portal_ui"}
    )
    
    # If negative, record a specific flag for triage
    if not is_positive:
        client.create_feedback(
            run_id=run_id,
            key="triage_required",
            score=1.0,
            comment="User reported unhelpful response; flagged for human inspection."
        )
    
    status_label = "POSITIVE (+1)" if is_positive else "NEGATIVE (0)"
    print(f" -> Feedback recorded: {status_label} | Comment: '{user_comment}'")


def route_to_annotation_queue(client: Client, run_id: str):
    """
    Step 3: Route problematic or low-confidence runs to an Annotation Queue.
    Human annotators/engineers review these queues in the LangSmith UI.
    """
    print(f"\n[3. ANNOTATION QUEUE] Triaging Run ID [{run_id[:8]}...] for human review...")

    # Find existing queue or create a new one
    queues = list(client.list_annotation_queues(name=QUEUE_NAME))
    if queues:
        queue = queues[0]
        print(f" -> Reusing existing queue: '{queue.name}' (ID: {queue.id})")
    else:
        queue = client.create_annotation_queue(
            name=QUEUE_NAME,
            description="Queue for reviewing flagged customer queries and negative feedback traces."
        )
        print(f" -> Created new annotation queue: '{queue.name}' (ID: {queue.id})")

    # Add the run into the queue
    client.add_runs_to_annotation_queue(
        queue_id=queue.id,
        run_ids=[run_id]
    )
    print(f" -> Successfully routed run to queue '{queue.name}'!")
    print(f" -> Annotators can inspect this at: https://smith.langchain.com -> Annotation Queues")


def main():
    if not os.getenv("LANGCHAIN_API_KEY"):
        print("[ERROR] LANGCHAIN_API_KEY is required.")
        return

    client = Client()

    # Scenario A: Good interaction -> Positive Feedback
    run_id_good, _ = run_application_and_capture_id("How do I reset my password?")
    record_user_feedback(client, run_id_good, is_positive=True, user_comment="Fast and accurate!")

    # Scenario B: Problematic interaction -> Negative Feedback & Route to Queue
    run_id_bad, _ = run_application_and_capture_id("My billing charge of $999.00 on invoice #INV-492 is wrong!")
    record_user_feedback(client, run_id_bad, is_positive=False, user_comment="Didn't answer why the charge was made.")
    route_to_annotation_queue(client, run_id_bad)

    client.flush()
    print("\n[SUCCESS] Production feedback & annotation triage workflow completed!")

if __name__ == "__main__":
    main()
