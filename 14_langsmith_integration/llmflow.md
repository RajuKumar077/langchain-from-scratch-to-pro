**complete Mermaid flowchart with wrapped text, shorter node labels, and improved spacing** to prevent text from being cut off.

I've divided the content into **4 diagrams** so they remain readable instead of squeezing everything into one massive flowchart.

**Important:** Copy the code directly into Mermaid Live Editor. Start with `%%{init...` and do not include Markdown backticks.

## 1. LLM Evaluation: The Big Picture

```mermaid
%%{init: {"flowchart": {"htmlLabels": true, "nodeSpacing": 60, "rankSpacing": 80, "padding": 25, "useMaxWidth": true}, "themeVariables": {"fontSize": "16px"}}}%%
flowchart TD
    A["Traditional Software Tests<br/>Expected to pass 100%"]
    B["LLM Application Tests<br/>Cannot guarantee perfect results"]
    C["Why?<br/>Model Drift and Changing Data"]
    D["Measure Regression<br/>Compare against a baseline"]
    
    A --> B
    B --> C
    C --> D

    D --> E["STAGE 1<br/>DESIGN"]
    E --> F["Build a Self-Corrective<br/>RAG Application"]
    
    F --> G["STAGE 2<br/>OFFLINE EVALUATION"]
    G --> H["Test with Datasets<br/>and Known Answers"]
    
    H --> I["STAGE 3<br/>PRODUCTION MONITORING"]
    I --> J["Monitor Real Users<br/>and Application Traces"]
    
    J --> K["Find Errors<br/>and Collect Feedback"]
    K --> L["Add Production Errors<br/>to the Evaluation Dataset"]
    L --> H
    
    K --> M["Improve the Application<br/>and Redesign"]
    M --> F

    classDef design fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef offline fill:#dcfce7,stroke:#16a34a,color:#111827
    classDef production fill:#fef3c7,stroke:#d97706,color:#111827
    classDef feedback fill:#f3e8ff,stroke:#9333ea,color:#111827

    class E,F design
    class G,H offline
    class I,J production
    class K,L,M feedback
```

## 2. Stage 1: Self-Corrective RAG

This diagram shows how an application retrieves documents, grades them, generates an answer, and falls back to web search when needed.

```mermaid
%%{init: {"flowchart": {"htmlLabels": true, "nodeSpacing": 55, "rankSpacing": 75, "padding": 25, "useMaxWidth": true}, "themeVariables": {"fontSize": "16px"}}}%%
flowchart TD
    A(["User Question"])
    B["Route the Question<br/>Choose a Retrieval Method"]
    
    C["Vector Store<br/>Search Internal Documents"]
    D["Web Search<br/>Search External Information"]
    
    E["Retrieve Documents"]
    F["Grade Documents<br/>Are They Relevant?"]
    
    G{"Relevant<br/>Documents Found?"}
    
    H["Generate an Answer<br/>Using Retrieved Context"]
    I["Check the Answer<br/>for Hallucinations"]
    J["Check Answer<br/>Relevance to the Question"]
    
    K{"Answer Is<br/>Useful and Supported?"}
    L(["Return Answer"])
    
    M["Transform the Query<br/>Rewrite the Question"]
    N["Search the Web<br/>for Better Information"]
    
    O["Pydantic Binary Grader<br/>Returns yes or no"]
    P["Use Temperature 0<br/>for Consistent Grading"]
    
    A --> B
    B --> C
    B --> D
    C --> E
    D --> E
    E --> F
    F --> G
    
    G -- Yes --> H
    H --> I
    I --> J
    J --> K
    
    K -- Yes --> L
    K -- No --> M
    G -- No --> M
    
    M --> N
    N --> H
    
    O -. "Supports grading" .-> F
    O -. "Supports grading" .-> I
    O -. "Supports grading" .-> J
    P -. "Controls randomness" .-> O

    classDef process fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef decision fill:#fef3c7,stroke:#d97706,color:#111827
    classDef fallback fill:#f3e8ff,stroke:#9333ea,color:#111827
    classDef result fill:#dcfce7,stroke:#16a34a,color:#111827

    class B,C,D,E,F,H,I,J process
    class G,K decision
    class M,N fallback
    class L result
```

**Key takeaway:** A binary grader returns a clear `yes` or `no`, allowing application code to make routing decisions. Setting temperature to `0` reduces randomness but does not guarantee identical results in every environment.

## 3. Stage 2: Offline Evaluation and Agent Testing

This diagram covers dataset creation, evaluator types, judge improvement, regression testing, and agent evaluation.

```mermaid
%%{init: {"flowchart": {"htmlLabels": true, "nodeSpacing": 55, "rankSpacing": 80, "padding": 25, "useMaxWidth": true}, "themeVariables": {"fontSize": "15px"}}}%%
flowchart TD
    A(["Offline Evaluation<br/>Known Answers Available"])
    
    A --> B["Build an Evaluation Dataset"]
    
    B --> C["Manual Examples<br/>10 to 50 Cases"]
    B --> D["Application Logs<br/>Real Previous Inputs"]
    B --> E["Synthetic Examples<br/>Generated Test Cases"]
    
    C --> F["Choose Dataset Format"]
    D --> F
    E --> F
    
    F --> G["KV Dataset<br/>Flexible Key-Value Examples"]
    F --> H["LLM Dataset<br/>Model-Generated Examples"]
    F --> I["Chat Dataset<br/>Conversation Examples"]
    
    G --> J["Run Evaluators"]
    H --> J
    I --> J
    
    J --> K["Heuristic Evaluator<br/>Hardcoded Rules"]
    J --> L["Human Evaluation<br/>Manual Annotation"]
    J --> M["LLM-as-a-Judge<br/>Prompt-Based Grading"]
    
    K --> N["Reference-Free Checks<br/>Example: Valid JSON"]
    K --> O["Reference-Based Checks<br/>Example: Answer Accuracy"]
    
    L --> P["Annotation Queues<br/>Human Review"]
    
    M --> Q["Define Clear Grading Criteria"]
    Q --> R["Avoid Vague Scores<br/>Such as 0 to 10"]
    
    R --> S["Judge Scores the Output"]
    S --> T["Human Reviews the Score"]
    T --> U["Human Corrects the Judgment"]
    U --> V["Store Corrected Examples"]
    V --> W["Add Examples to the<br/>Judge System Prompt"]
    W --> S
    
    N --> X["Compare Application Versions"]
    O --> X
    P --> X
    W --> X
    
    X --> Y["Pairwise Comparison<br/>Compare Version A with B"]
    X --> Z["Regression Testing<br/>Compare Against Baseline"]
    
    Y --> AA["Ranked Preference<br/>Which Answer Is Better?"]
    Z --> AB["Identify Improvements<br/>and Regressions"]
    
    AA --> AC["Inspect Individual Examples"]
    AB --> AC
    
    AC --> AD["Isolate App Version<br/>and Evaluation Examples"]
    
    AD --> AE(["Decide What to Improve"])

    classDef dataset fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef evaluator fill:#dcfce7,stroke:#16a34a,color:#111827
    classDef judge fill:#f3e8ff,stroke:#9333ea,color:#111827
    classDef compare fill:#fef3c7,stroke:#d97706,color:#111827

    class B,C,D,E,F,G,H,I dataset
    class J,K,L,M,N,O,P evaluator
    class Q,R,S,T,U,V,W judge
    class X,Y,Z,AA,AB,AC,AD,AE compare
```

### Agent evaluation: three levels

Agents are nondeterministic, so evaluating only their final answers may miss important problems.

```mermaid
%%{init: {"flowchart": {"htmlLabels": true, "nodeSpacing": 55, "rankSpacing": 80, "padding": 25, "useMaxWidth": true}, "themeVariables": {"fontSize": "15px"}}}%%
flowchart TD
    A(["Agent Evaluation"])
    
    A --> B["Level 1<br/>Response Evaluation"]
    A --> C["Level 2<br/>Single-Step Evaluation"]
    A --> D["Level 3<br/>Trajectory Evaluation"]
    
    B --> E["Black-Box Assessment<br/>Input and Final Answer"]
    E --> F["LLM-as-a-Judge"]
    F --> G["Compare Answer with<br/>Reference Answer"]
    G --> H["Example Evaluator<br/>rag-answer-vs-reference"]
    
    C --> I["Inspect One Decision<br/>and Its Tool Call"]
    I --> J["Binary Score<br/>Was the Tool Correct?"]
    
    D --> K["Inspect the Full Sequence<br/>of Tool Calls"]
    K --> L["Extract Tool Calls<br/>from Messages"]
    
    L --> M["Matcher 1<br/>All Expected Calls<br/>in Any Order"]
    L --> N["Matcher 2<br/>All Expected Calls<br/>in Correct Order"]
    L --> O["Matcher 3<br/>Exact Ordered Match<br/>No Extra Calls"]
    
    H --> P["Repeat Evaluation<br/>Across Multiple Runs"]
    J --> P
    M --> P
    N --> P
    O --> P
    
    P --> Q["Use num_repetitions = 3<br/>to Account for Variability"]
    Q --> R(["Review Results"])

    classDef level fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef evaluation fill:#dcfce7,stroke:#16a34a,color:#111827
    classDef repeat fill:#fef3c7,stroke:#d97706,color:#111827

    class B,C,D level
    class E,F,G,H,I,J,K,L,M,N,O evaluation
    class P,Q,R repeat
```

**Remember the difference:**

* **Response evaluation:** Is the final answer good?
* **Single-step evaluation:** Was the tool choice correct?
* **Trajectory evaluation:** Did the agent follow the correct sequence of actions?

The evaluator `rag-answer-vs-reference` is an example of an evaluator configuration, not a universal requirement. In LangSmith, the exact evaluator name and availability depend on the project and configuration.

## 4. Stage 3: Production Monitoring and the Feedback Loop

Production monitoring usually has no known correct answer for every user request. Instead, you inspect traces, measure operational metrics, collect feedback, and apply suitable evaluators.

```mermaid
%%{init: {"flowchart": {"htmlLabels": true, "nodeSpacing": 55, "rankSpacing": 80, "padding": 25, "useMaxWidth": true}, "themeVariables": {"fontSize": "15px"}}}%%
flowchart TD
    A(["Deploy the LLM Application"])
    
    A --> B["Enable Tracing"]
    B --> C["Configure LangSmith<br/>Environment Variables"]
    
    C --> D["Capture Application Traces"]
    
    D --> E["Trace<br/>Overall Execution"]
    D --> F["Span<br/>Individual Execution Step"]
    
    E --> G["Monitor Operational Metrics"]
    F --> G
    
    G --> H["Trace Volume<br/>Number of Runs"]
    G --> I["Success and Failure<br/>Execution Outcomes"]
    G --> J["Latency<br/>Response Time"]
    G --> K["Token Usage<br/>and Cost"]
    
    D --> L["Collect User Feedback"]
    
    L --> M["Thumbs Up or Down"]
    L --> N["Inline Annotation"]
    L --> O["Human Annotation Queue"]
    
    D --> P["Evaluate Traces with an LLM Judge"]
    P --> Q["Check for Hallucinations"]
    P --> R["Check for Toxicity"]
    
    H --> S["Choose Evaluation Strategy"]
    I --> S
    J --> S
    K --> S
    M --> S
    N --> S
    O --> S
    Q --> S
    R --> S
    
    S --> T{"Are Reference<br/>Labels Available?"}
    
    T -- Yes --> U["Custom Heuristic Evaluator<br/>Compare Against Labels"]
    T -- No --> V["LLM-as-a-Judge<br/>Apply Explicit Criteria"]
    
    U --> W["Identify Production Errors"]
    V --> W
    
    W --> X["Add Errors and Corrections<br/>to the Offline Dataset"]
    X --> Y["Rerun Offline Evaluations"]
    
    Y --> Z["Improve the Application"]
    Z --> AA["Release to a Small<br/>Beta User Group"]
    
    AA --> AB["Collect Feedback<br/>and Build Ground Truth"]
    AB --> AC["Expand the Release<br/>When Results Are Acceptable"]
    
    AC --> AD(["Continue Monitoring"])
    AD --> D

    classDef production fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef metrics fill:#dcfce7,stroke:#16a34a,color:#111827
    classDef feedback fill:#f3e8ff,stroke:#9333ea,color:#111827
    classDef decision fill:#fef3c7,stroke:#d97706,color:#111827

    class A,B,C,D,E,F production
    class G,H,I,J,K metrics
    class L,M,N,O,P,Q,R feedback
    class S,T,U,V,W,X,Y,Z,AA,AB,AC,AD decision
```

### Production tracing setup

For a typical LangSmith tracing setup, the environment variables look like this:

```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_API_KEY="your_langsmith_api_key"
```

Set your API key securely in your environment rather than hardcoding it in source code. You may also need to configure the project name and other settings for your application.

### The complete feedback loop

The most important concept across all four diagrams is:

**Design → Test → Deploy → Monitor → Learn → Improve → Test Again**

