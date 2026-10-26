%%{init: {"flowchart": {"htmlLabels": true, "nodeSpacing": 35, "rankSpacing": 55, "padding": 15, "useMaxWidth": true}, "themeVariables": {"fontSize": "13px"}}}%%

flowchart TD

    START(["LLM APPLICATION<br/>EVALUATION"])

    START --> WHY["Why Evaluate LLMs?<br/>Model behavior can change"]
    WHY --> DRIFT["Model Drift<br/>Changing data or model updates"]
    DRIFT --> BASELINE["Compare Against a Baseline<br/>Measure regressions, not perfection"]

    BASELINE --> DESIGN_START
    BASELINE --> OFFLINE_START
    BASELINE --> PROD_START

    %% =========================================
    %% STAGE 1: DESIGN AND SELF-CORRECTIVE RAG
    %% =========================================

    subgraph DESIGN["STAGE 1: DESIGN - SELF-CORRECTIVE RAG"]
        direction TB

        DESIGN_START(["Design the Application"])

        R1["Receive User Question"]
        R2["Route the Question<br/>Choose a retrieval method"]

        R3["Vector Store<br/>Search internal documents"]
        R4["Web Search<br/>Search external information"]

        R5["Retrieve Documents"]
        R6["Grade Documents<br/>Are they relevant?"]
        R7{"Relevant<br/>documents found?"}

        R8["Generate Answer<br/>Use retrieved context"]
        R9["Check for Hallucinations<br/>Is the answer supported?"]
        R10["Check Answer Relevance<br/>Does it answer the question?"]
        R11{"Answer is<br/>useful and supported?"}

        R12(["Return Final Answer"])

        R13["Transform the Query<br/>Rewrite the question"]
        R14["Search the Web<br/>Find better information"]

        R15["Binary Grader<br/>Pydantic yes/no score"]
        R16["Temperature 0<br/>Reduces randomness"]

        DESIGN_START --> R1
        R1 --> R2

        R2 --> R3
        R2 --> R4

        R3 --> R5
        R4 --> R5

        R5 --> R6
        R6 --> R7

        R7 -- Yes --> R8
        R7 -- No --> R13

        R8 --> R9
        R9 --> R10
        R10 --> R11

        R11 -- Yes --> R12
        R11 -- No --> R13

        R13 --> R14
        R14 --> R8

        R15 -. "Grades relevance" .-> R6
        R15 -. "Grades answer quality" .-> R9
        R15 -. "Grades answer quality" .-> R10
        R16 -. "Configures grader" .-> R15
    end

    %% =========================================
    %% STAGE 2: OFFLINE EVALUATION
    %% =========================================

    subgraph OFFLINE["STAGE 2: OFFLINE EVALUATION"]
        direction TB

        OFFLINE_START(["Ground Truth Available"])

        E1["Create Evaluation Dataset"]

        E2["Manual Examples<br/>10 to 50 cases"]
        E3["Application Logs<br/>Real previous inputs"]
        E4["Synthetic Examples<br/>Generated test cases"]

        E5["Dataset Formats"]
        E6["KV Dataset<br/>Flexible key-value examples"]
        E7["LLM Dataset<br/>Model-generated examples"]
        E8["Chat Dataset<br/>Conversation examples"]

        E9["Run Evaluators"]

        E10["Heuristic Evaluator<br/>Hardcoded rules"]
        E11["Human Evaluator<br/>Manual annotation"]
        E12["LLM-as-a-Judge<br/>Prompt-based grading"]

        E13["Reference-Free Check<br/>Example: valid JSON"]
        E14["Reference-Based Check<br/>Example: answer accuracy"]

        E15["Annotation Queue<br/>Humans review difficult cases"]

        E16["Define Clear Judge Criteria"]
        E17["Avoid Vague Scores<br/>Such as 0 to 10"]

        E18["Judge Scores the Output"]
        E19["Human Reviews the Score"]
        E20["Human Corrects the Judgment"]
        E21["Store Corrected Examples"]
        E22["Add Examples to Judge<br/>System Prompt"]

        E23["Compare Application Versions"]

        E24["Pairwise Comparison<br/>Version A versus Version B"]
        E25["Regression Testing<br/>Compare against baseline"]

        E26["Ranked Preference<br/>Which answer is better?"]
        E27["Identify Improvements<br/>and Regressions"]

        E28["Inspect Individual Examples"]
        E29["Isolate App Version<br/>and test examples"]

        E30(["Decide What to Improve"])

        OFFLINE_START --> E1

        E1 --> E2
        E1 --> E3
        E1 --> E4

        E2 --> E5
        E3 --> E5
        E4 --> E5

        E5 --> E6
        E5 --> E7
        E5 --> E8

        E6 --> E9
        E7 --> E9
        E8 --> E9

        E9 --> E10
        E9 --> E11
        E9 --> E12

        E10 --> E13
        E10 --> E14

        E11 --> E15

        E12 --> E16
        E16 --> E17
        E17 --> E18
        E18 --> E19
        E19 --> E20
        E20 --> E21
        E21 --> E22
        E22 --> E18

        E13 --> E23
        E14 --> E23
        E15 --> E23
        E22 --> E23

        E23 --> E24
        E23 --> E25

        E24 --> E26
        E25 --> E27

        E26 --> E28
        E27 --> E28

        E28 --> E29
        E29 --> E30
    end

    %% =========================================
    %% AGENT EVALUATION: THREE LEVELS
    %% =========================================

    subgraph AGENTS["AGENT EVALUATION: THREE LEVELS"]
        direction TB

        AG_START(["Evaluate an Agent"])

        AG1["Level 1<br/>Response Evaluation"]
        AG2["Level 2<br/>Single-Step Evaluation"]
        AG3["Level 3<br/>Trajectory Evaluation"]

        AG4["Inspect Input and<br/>Final Answer"]
        AG5["LLM-as-a-Judge"]
        AG6["Compare Answer<br/>Against Reference"]
        AG7["Example Evaluator<br/>rag-answer-vs-reference"]

        AG8["Inspect One Decision<br/>and Its Tool Call"]
        AG9["Binary Score<br/>Was the tool choice correct?"]

        AG10["Inspect the Full Sequence<br/>of Tool Calls"]
        AG11["Extract Tool Calls<br/>from Messages"]

        AG12["Matcher: Any Order<br/>All expected calls included"]
        AG13["Matcher: In Order<br/>Expected calls in sequence"]
        AG14["Matcher: Exact Match<br/>Correct order, no extras"]

        AG15["Repeat Evaluation<br/>num_repetitions = 3"]
        AG16(["Review Agent Results"])

        AG_START --> AG1
        AG_START --> AG2
        AG_START --> AG3

        AG1 --> AG4
        AG4 --> AG5
        AG5 --> AG6
        AG6 --> AG7

        AG2 --> AG8
        AG8 --> AG9

        AG3 --> AG10
        AG10 --> AG11

        AG11 --> AG12
        AG11 --> AG13
        AG11 --> AG14

        AG7 --> AG15
        AG9 --> AG15
        AG12 --> AG15
        AG13 --> AG15
        AG14 --> AG15

        AG15 --> AG16
    end

    %% =========================================
    %% STAGE 3: PRODUCTION MONITORING
    %% =========================================

    subgraph PRODUCTION["STAGE 3: PRODUCTION MONITORING"]
        direction TB

        PROD_START(["Deploy Application"])

        P1["Enable LangSmith Tracing"]
        P2["Configure Environment Variables"]
        P3["Capture Application Traces"]

        P4["Trace<br/>Overall execution"]
        P5["Span<br/>Individual execution step"]

        P6["Monitor Operational Metrics"]

        P7["Trace Volume<br/>Number of runs"]
        P8["Success and Failure<br/>Execution outcomes"]
        P9["Latency<br/>Response time"]
        P10["Tokens and Cost<br/>Resource usage"]

        P11["Collect User Feedback"]
        P12["Thumbs Up or Down"]
        P13["Inline Annotation"]
        P14["Human Annotation Queue"]

        P15["Run LLM Judge on Traces"]
        P16["Check for Hallucinations"]
        P17["Check for Toxicity"]

        P18{"Reference Labels<br/>Available?"}

        P19["Heuristic Evaluator<br/>Compare against labels"]
        P20["LLM-as-a-Judge<br/>Apply explicit criteria"]

        P21["Identify Production Errors"]
        P22["Add Errors and Corrections<br/>to Offline Dataset"]

        P23["Rerun Offline Evaluations"]
        P24["Improve the Application"]

        P25["Release to a Small<br/>Beta User Group"]
        P26["Collect Feedback<br/>Build Ground Truth"]

        P27["Expand Release<br/>When results are acceptable"]

        P28(["Continue Monitoring"])

        PROD_START --> P1
        P1 --> P2
        P2 --> P3

        P3 --> P4
        P3 --> P5

        P4 --> P6
        P5 --> P6

        P6 --> P7
        P6 --> P8
        P6 --> P9
        P6 --> P10

        P3 --> P11
        P11 --> P12
        P11 --> P13
        P11 --> P14

        P3 --> P15
        P15 --> P16
        P15 --> P17

        P7 --> P18
        P8 --> P18
        P9 --> P18
        P10 --> P18

        P18 -- Yes --> P19
        P18 -- No --> P20

        P12 --> P21
        P13 --> P21
        P14 --> P21
        P16 --> P21
        P17 --> P21
        P19 --> P21
        P20 --> P21

        P21 --> P22
        P22 --> P23
        P23 --> P24

        P24 --> P25
        P25 --> P26
        P26 --> P27
        P27 --> P28

        P28 --> P3
    end

    %% =========================================
    %% COMPLETE FEEDBACK LOOP
    %% =========================================

    R12 -. "Evaluate application behavior" .-> OFFLINE_START

    E30 -. "Validate improvements" .-> AG_START

    AG16 -. "Validate agent behavior" .-> PROD_START

    P22 -. "Add production failures<br/>to test dataset" .-> E1

    P24 -. "Verify fixes against baseline" .-> E23

    P24 -. "Redesign and improve" .-> R1

    %% =========================================
    %% STYLING
    %% =========================================

    classDef start fill:#1e293b,stroke:#0f172a,color:#ffffff
    classDef design fill:#dbeafe,stroke:#2563eb,color:#111827
    classDef offline fill:#dcfce7,stroke:#16a34a,color:#111827
    classDef agent fill:#f3e8ff,stroke:#9333ea,color:#111827
    classDef production fill:#fef3c7,stroke:#d97706,color:#111827
    classDef decision fill:#ffedd5,stroke:#ea580c,color:#111827

    class START start

    class DESIGN_START,R1,R2,R3,R4,R5,R6,R8,R9,R10,R12,R13,R14,R15,R16 design

    class OFFLINE_START,E1,E2,E3,E4,E5,E6,E7,E8,E9,E10,E11,E12,E13,E14,E15,E16,E17,E18,E19,E20,E21,E22,E23,E24,E25,E26,E27,E28,E29,E30 offline

    class AG_START,AG1,AG2,AG3,AG4,AG5,AG6,AG7,AG8,AG9,AG10,AG11,AG12,AG13,AG14,AG15,AG16 agent

    class PROD_START,P1,P2,P3,P4,P5,P6,P7,P8,P9,P10,P11,P12,P13,P14,P15,P16,P17,P19,P20,P21,P22,P23,P24,P25,P26,P27,P28 production

    class R7,R11,E18,P18 decision
    