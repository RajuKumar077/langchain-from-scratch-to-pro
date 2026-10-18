# Module 14: LangSmith Tracing, Evaluation & Deployment (100% Local Ollama + Docker Setup)

Welcome to **Module 14**!

This module demonstrates how to integrate **LangSmith** with your **100% local infrastructure**:
- 🤖 **Local LLM**: Ollama (`qwen3:8b`)
- 🐳 **Local Database**: PostgreSQL + PGVector Docker container
- 📊 **Observability & Tracing**: LangSmith (for prompt monitoring, step-by-step trace trees, evaluation, and latency tracking)

> 💡 **No OpenAI or Anthropic API Keys Required!**  
> All model inferences run completely locally on your hardware. LangSmith only receives lightweight trace metadata (prompts, inputs, outputs, execution times) so you can inspect and debug your runs visually in the online dashboard.

---

## 📁 Directory Structure

```
14_langsmith_integration/
├── .env.example                     # Environment template for LangSmith & Local Ollama
├── README.md                        # Complete step-by-step setup and deployment guide
├── config.py                        # Central environment checker & configuration utility
├── 01_basic_tracing.py              # Local Ollama LCEL tracing, metadata tagging & @traceable decorator
├── 02_langgraph_tracing.py          # Tracing multi-node LangGraph agents and conditional edges
├── 03_datasets_and_evaluation.py    # Ground-truth datasets, custom evaluators & automated benchmarking
├── 04_langgraph_cloud_deployment.py # Production graph entrypoint optimized for server deployment
└── langgraph.json                   # LangGraph Cloud / LangSmith Deployment manifest file
```

---

## 🚀 Quick Start Guide

### Step 1: Copy Environment Variables

Create your local `.env` file from `.env.example`:

```bash
cp .env.example .env
```

Set your free **LangSmith API Key** (from [https://smith.langchain.com](https://smith.langchain.com)):

```env
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=lsv2_pt_your_actual_api_key_here
LANGCHAIN_PROJECT=langchain-book-demo
LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# Local Ollama settings
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=qwen3:8b
```

### Step 2: Verify Setup

Run the configuration check script:

```bash
r:\LangChain-Book\.venv\Scripts\python.exe config.py
```

---

## 🎓 Lessons Breakdown

### Lesson 01: Basic Tracing with Local Ollama (`01_basic_tracing.py`)
- Executes LCEL chain using your local `qwen3:8b` model via `from langchain_ollama import ChatOllama`.
- Automatically logs run metadata, model latency, and prompt traces to LangSmith.
- Uses `@traceable` decorator from `langsmith` for custom Python functions.

Run:
```bash
r:\LangChain-Book\.venv\Scripts\python.exe 01_basic_tracing.py
```

### Lesson 02: Tracing LangGraph Workflows (`02_langgraph_tracing.py`)
- Traces a multi-node routing agent in LangGraph.
- Displays state transitions, conditional routing decisions, and node execution timings.

Run:
```bash
r:\LangChain-Book\.venv\Scripts\python.exe 02_langgraph_tracing.py
```

### Lesson 03: Datasets & Evaluation (`03_datasets_and_evaluation.py`)
- Creates a benchmark dataset on LangSmith (`langsmith.Client`).
- Evaluates your local `qwen3:8b` responses against ground-truth answers using custom evaluators.

Run:
```bash
r:\LangChain-Book\.venv\Scripts\python.exe 03_datasets_and_evaluation.py
```

### Lesson 04 & Deployment (`04_langgraph_cloud_deployment.py`)
- Prepares your local LangGraph agent for production deployment using `langgraph.json`.

---

## 📊 Viewing Traces on LangSmith UI

1. Open [https://smith.langchain.com](https://smith.langchain.com).
2. Open Project **`langchain-book-demo`**.
3. View the full trace waterfall chart showing node steps, prompts, and local model responses!
