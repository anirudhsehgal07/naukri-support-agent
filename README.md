# Naukri.com Domain Support Agent (LangGraph)

**Capstone Track:** Recruitment & HR (Naukri.com)  
**Project:** Naukri.com Domain Support Agent using LangGraph  
**Institution:** Masai School

---

## 1. Project Overview

The Naukri.com Domain Support Agent is a recruitment support system built using Python, LangGraph, Retrieval-Augmented Generation (RAG), ChromaDB, FastAPI and FastMCP.

The agent helps candidates and recruiters by:

- Answering recruitment policy questions using a local knowledge base.
- Checking job application status using application IDs.
- Calculating application escalation scores.
- Maintaining conversation history across multiple interactions.
- Detecting prompt injection attempts.
- Masking fixed-format phone numbers.
- Validating structured responses.
- Supporting workflow checkpointing, retries and timeouts.
- Exposing application lookup functionality through MCP.

The project uses deterministic `MOCK_LLM` mode for grounded generation and evaluation. It does not require an external LLM API key.

## 2. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| LangGraph | Agent workflow orchestration |
| SentenceTransformers | Text embeddings |
| ChromaDB | Vector database |
| all-MiniLM-L6-v2 | Embedding model |
| FastAPI | Backend API |
| Pydantic | Data and response validation |
| FastMCP | MCP server and client |
| SQLite | LangGraph checkpointing |
| JSON | Conversation history |
| JSONL | Structured logging |
| asyncio | Timeout and asynchronous execution |

## 3. Project Structure

```text
naukri-support-agent/
│
├── dataset.py
├── create_knowledge_base.py
├── chunking.py
├── build_vector_db.py
├── test_vector_db.py
├── calibrate_threshold.py
├── rag_engine.py
├── evaluate_chunking.py
├── application_lookup.py
├── agent_workflow.py
├── conversation_manager.py
├── response_validator.py
├── input_guardrails.py
├── api.py
├── logging_utils.py
├── rag_evaluation.py
├── fastmcp_server.py
├── fastmcp_client.py
├── checkpoint_demo.py
├── reliability_demo.py
│
├── knowledge_base/
│   ├── eligibility.txt
│   ├── interview_scheduling.txt
│   ├── offer_negotiation.txt
│   ├── background_verification.txt
│   ├── notice_period.txt
│   ├── referral_bonus.txt
│   ├── internal_transfer.txt
│   ├── probation.txt
│   ├── remote_work.txt
│   ├── diversity_hiring.txt
│   ├── exit_interview.txt
│   └── data_retention.txt
│
├── chroma_db/
├── logs/
│   └── agent_logs.jsonl
│
├── calibration_results.json
├── rag_evaluation_results.json
├── conversation_history.json
├── checkpoints.sqlite
├── requirements.txt
└── README.md
```

The ChromaDB directory is generated locally by running `build_vector_db.py`. The checkpoint database and logs are generated during execution.

---

# PART 1 — DATASET DESIGN & RAG CORE

## 4. Dataset Generation

**File:** `dataset.py`

A deterministic dataset of 50 job application records was generated.

### Dataset design choices

| Setting | Value |
|---|---:|
| Total records | 50 |
| Dataset seed | 2026 |
| Priority seed | 42 |
| Number of categories | 5 |
| Number of statuses | 5 |
| Priority review records | 12 |
| Priority review percentage | 24% |

### Category distribution

| Category | Records |
|---|---:|
| Software Engineer | 10 |
| Data Analyst | 10 |
| Product Manager | 10 |
| HR Executive | 10 |
| Sales Associate | 10 |

### Status distribution

| Status | Records |
|---|---:|
| Applied | 10 |
| Screening | 10 |
| Interview Scheduled | 10 |
| Offered | 10 |
| Rejected | 10 |

### Salary ranges

| Category | Annual salary range (INR) |
|---|---:|
| Software Engineer | ₹4,00,000–₹18,00,000 |
| Data Analyst | ₹3,50,000–₹12,00,000 |
| Product Manager | ₹7,00,000–₹20,00,000 |
| HR Executive | ₹2,50,000–₹8,00,000 |
| Sales Associate | ₹2,40,000–₹9,00,000 |

The salary ranges were selected to provide different role-based annual compensation bands for the five recruitment categories.

Equal category and status weights were used to produce a balanced dataset. Fixed random seeds make the generated dataset reproducible.

Each record contains:

- `record_id`
- `category`
- `status`
- `expected_salary_inr`
- `days_since_created`
- `flagged_priority_review`

Run:

```bash
python dataset.py
```

The script prints category counts, status counts and the priority-review percentage.

## 5. Knowledge Base Creation

**Files:** `create_knowledge_base.py`, `knowledge_base/`

The knowledge base contains 12 documents, each written in 2–5 sentences.

The required topics are:

1. Job application eligibility criteria
2. Interview scheduling process
3. Offer negotiation policy
4. Background verification process
5. Notice period policy
6. Referral bonus policy
7. Internal transfer eligibility
8. Probation period policy
9. Remote work eligibility
10. Diversity hiring guidelines
11. Exit interview process
12. Applicant data retention policy

These documents provide the source information used by the RAG system.

Run:

```bash
python create_knowledge_base.py
```

## 6. Document Chunking and Vector Database

**Files:** `chunking.py`, `build_vector_db.py`

Two chunking strategies were implemented.

### Fixed-size chunking with overlap

- Chunk size: 150
- Overlap: 30

### Sentence-based chunking

Documents are split into sentence-level chunks.

Both strategies use the SentenceTransformer model:

`all-MiniLM-L6-v2`

ChromaDB stores the embeddings in separate collections:

- `fixed_size_overlap`
- `sentence_based`

The generated collections contain:

| Collection | Chunks |
|---|---:|
| Fixed-size overlap | 37 |
| Sentence-based | 36 |

Run:

```bash
python build_vector_db.py
```

To test the collections:

```bash
python test_vector_db.py
```

## 7. Grounded RAG Generation

**File:** `rag_engine.py`

The RAG system:

1. Receives a user question.
2. Retrieves relevant chunks from ChromaDB.
3. Checks the retrieval distance against a calibrated threshold.
4. Uses retrieved context to generate an answer.
5. Refuses unsupported questions.

The selected cosine-distance threshold is:

`0.63`

Lower cosine distance indicates greater similarity.

### Threshold calibration

The threshold was selected using measured in-scope and out-of-scope retrieval distances.

| Collection | In-scope distance range | Out-of-scope distance range |
|---|---:|---:|
| Fixed-size overlap | 0.2918–0.4021 | 0.7960–0.9462 |
| Sentence-based | 0.2006–0.3891 | 0.8656–0.9574 |

For the sentence-based collection:

- Maximum observed in-scope distance: 0.3891
- Minimum observed out-of-scope distance: 0.8656

The midpoint is:

`(0.3891 + 0.8656) / 2 = 0.62735`

The selected threshold was rounded to `0.63`.

This threshold lies between the observed in-scope and out-of-scope distance clusters.

### MOCK_LLM mode

The project uses deterministic MOCK_LLM generation. Answers are constructed from retrieved knowledge-base context rather than external LLM inference.

Out-of-scope questions trigger a fallback response when retrieval confidence is insufficient.

## 8. Chunking Strategy Evaluation

**File:** `evaluate_chunking.py`

Both chunking strategies were compared using the same five queries.

The evaluation uses document-level Precision@3 and Recall@3. Retrieved chunks are mapped to their source documents, and duplicate source documents are removed before calculating the metrics.

### Evaluation queries

| No. | Query | Relevant document |
|---|---|---|
| 1 | How does the employee referral bonus work? | referral_bonus.txt |
| 2 | Can employees work remotely? | remote_work.txt |
| 3 | What happens during an exit interview? | exit_interview.txt |
| 4 | What qualifications must applicants have? | eligibility.txt |
| 5 | How can an employee transfer to another department? | internal_transfer.txt |

Each query has one expected relevant source document.

### Metric formulas

**Precision@3**

`Relevant retrieved documents / 3`

**Recall@3**

`Relevant retrieved documents / Total relevant documents`

For each query, one relevant document was retrieved among the top three unique source documents by both strategies.

### Per-query results: Fixed-size with overlap

| Query | Relevant found | Precision@3 calculation | Recall@3 calculation |
|---|---:|---|---|
| Referral bonus | 1 | 1/3 = 0.333 | 1/1 = 1.000 |
| Remote work | 1 | 1/3 = 0.333 | 1/1 = 1.000 |
| Exit interview | 1 | 1/3 = 0.333 | 1/1 = 1.000 |
| Eligibility | 1 | 1/3 = 0.333 | 1/1 = 1.000 |
| Internal transfer | 1 | 1/3 = 0.333 | 1/1 = 1.000 |

### Per-query results: Sentence-based

| Query | Relevant found | Precision@3 calculation | Recall@3 calculation |
|---|---:|---|---|
| Referral bonus | 1 | 1/3 = 0.333 | 1/1 = 1.000 |
| Remote work | 1 | 1/3 = 0.333 | 1/1 = 1.000 |
| Exit interview | 1 | 1/3 = 0.333 | 1/1 = 1.000 |
| Eligibility | 1 | 1/3 = 0.333 | 1/1 = 1.000 |
| Internal transfer | 1 | 1/3 = 0.333 | 1/1 = 1.000 |

### Average results

| Metric | Fixed-size overlap | Sentence-based |
|---|---:|---:|
| Average Precision@3 | 0.333 | 0.333 |
| Average Recall@3 | 1.000 | 1.000 |

### Recommendation

Sentence-based chunking is recommended because it preserves sentence boundaries and produces more meaningful text units.

Both strategies achieved equal measured Precision@3 and Recall@3. Therefore, the recommendation is based on chunk structure and readability rather than a demonstrated retrieval accuracy advantage.

Run:

```bash
python evaluate_chunking.py
```

---

# PART 2 — LANGGRAPH AGENT WITH TOOLS, MEMORY & GUARDRAILS

## 9. Application Status Lookup and Escalation Score

**File:** `application_lookup.py`

The function:

```python
check_job_application_status(record_id: str) -> dict
```

looks up an application using its record ID.

It returns:

- Application status
- Expected salary
- Escalation score
- Escalation requirement

### Escalation formula

```text
priority_score = 1 if flagged_priority_review else 0

recency_score = (30 - days_since_created) / 30

escalation_score =
    (0.60 * priority_score) +
    (0.40 * recency_score)
```

The recency score is normalized between 0 and 1 because application age ranges from 0 to 30 days.

### Escalation threshold

`ESCALATION_THRESHOLD = 0.60`

An application is escalated when:

`escalation_score >= 0.60`

### Threshold selection and justification

The threshold was selected based on the distribution of the generated dataset.

The dataset contains 50 application records:

- 12 records (24%) are flagged for priority review.
- 38 records (76%) are not flagged for priority review.

Based on the scoring formula:

- A priority-flagged record has a score between 0.60 and 1.00.
- A non-priority record has a score between 0.00 and 0.40.

Therefore, the threshold of 0.60 separates priority-flagged records from non-priority records in this dataset.

This ensures that all priority-flagged applications are escalated, while non-priority applications are not automatically escalated.

The 60% priority weight ensures that priority review has the strongest influence. The 40% recency weight adds urgency based on how recently the application was created.

The threshold is a deterministic policy choice based on the dataset distribution, not a statistically trained prediction threshold.

## 10. LangGraph Workflow

**File:** `agent_workflow.py`

The LangGraph agent contains four nodes:

1. `classify_intent`
2. `rag_node`
3. `lookup_node`
4. `response_node`

### Workflow routing

- Recruitment policy questions are sent to the RAG node.
- Application-status questions are sent to the lookup node.
- Both routes connect to the final response node.

The workflow uses a conditional edge to select the appropriate route.

### Workflow flow

```text
START
  |
  v
classify_intent
  |
  |-- RAG question --------> rag_node
  |
  |-- Application lookup --> lookup_node
                                  |
                                  v
                            response_node
                                  |
                                  v
                                 END
```

## 11. Conversation Memory

**File:** `conversation_manager.py`

Conversation history is stored in:

`conversation_history.json`

Implemented features:

- Save user messages.
- Save assistant responses.
- Load previous conversation messages.
- Support multi-turn conversations.
- Start a fresh conversation using `/new`.

## 12. Structured Response Validation

**File:** `response_validator.py`

Pydantic schemas validate the agent's responses.

The validation covers:

- Application responses
- RAG responses
- Error responses

The response node checks validation before returning the final answer.

Invalid field values are rejected according to the defined schema constraints.

## 13. Input Guardrails and Groundedness

**File:** `input_guardrails.py`

### Phone number masking

The input guardrail detects fixed-format Indian mobile numbers and masks them.

Example:

```text
Input: 9876543210
Output: ********10
```

The last two digits are retained while the preceding digits are masked.

### Prompt injection detection

The guardrail detects suspicious instructions such as:

- Ignore previous instructions.
- Reveal your system prompt.
- Bypass your rules.

Unsafe input is blocked before being processed by the agent.

### Output groundedness

The RAG system checks retrieval distance against the calibrated threshold.

If the retrieved context is insufficient, it refuses to provide an unsupported answer.

---

# PART 3 — EVALUATION, OBSERVABILITY & FASTAPI

## 14. FastAPI Deployment

**File:** `api.py`

The agent is exposed through FastAPI with Pydantic request and response models.

### API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Welcome message |
| POST | `/chat` | Send a question to the agent |
| GET | `/application/{record_id}` | Retrieve application status |

### Start the API

```bash
uvicorn api:api --reload
```

Open the interactive API documentation:

```text
http://127.0.0.1:8000/docs
```

### API testing

The following endpoints were tested successfully:

- `GET /`
- `GET /application/APP0001`
- `POST /chat`

All three returned HTTP 200 responses during successful tests.

The chat endpoint returned a grounded recruitment answer with successful response validation.

## 15. Structured Logging

**File:** `logging_utils.py`

Structured logs are written in JSONL format.

Log information includes:

- Timestamp
- Trace ID
- Masked question or request description
- Detected intent
- Request status
- Execution duration in milliseconds

Log file:

`logs/agent_logs.jsonl`

The input masking function is applied before the chat request is logged.

The chat, application lookup and root endpoints generate structured log entries. Each request receives a unique trace ID.

## 16. RAG Triad Evaluation

**File:** `rag_evaluation.py`

A test set of 15 queries was created.

It contains:

- 12 in-scope queries covering all required knowledge-base topics.
- 3 out-of-scope queries.

A prompt-guided deterministic MOCK_LLM judge evaluates three metrics:

1. Context relevance
2. Groundedness
3. Answer relevance

Each metric is scored for every query on a scale from 0 to 1.

### Evaluation results

| Metric | Average score |
|---|---:|
| Context relevance | 1.00 |
| Groundedness | 1.00 |
| Answer relevance | 0.87 |

All 15 queries received all three scores.

The results are saved in:

`rag_evaluation_results.json`

### Evaluation note

The judge uses a deterministic, prompt-guided mock scoring implementation. It does not perform genuine external LLM inference.

The out-of-scope queries receive positive evaluation scores when the system correctly refuses unsupported questions. This represents successful refusal behavior, not successful answering of out-of-scope questions.

Run:

```bash
python rag_evaluation.py
```

---

# PART 4 — RESILIENCE & INTEROPERABILITY

## 17. FastMCP Server and Client

**Files:** `fastmcp_server.py`, `fastmcp_client.py`

FastMCP exposes the application lookup function as an MCP tool.

Tool name:

`get_application_status`

The server uses local HTTP transport.

### Start the MCP server

Open a terminal and run:

```bash
python fastmcp_server.py
```

The server runs at:

```text
http://127.0.0.1:8001
```

The MCP endpoint is:

```text
http://127.0.0.1:8001/mcp
```

Keep the server terminal running.

### Run the separate MCP client

Open a second terminal and run:

```bash
python fastmcp_client.py
```

The client connects to the HTTP MCP endpoint and calls the application lookup tool for two application IDs.

### Test results

| Application ID | Status | Expected salary | Escalation score | Escalation |
|---|---|---:|---:|---|
| APP0001 | Applied | ₹11,14,000 | 0.28 | No |
| APP0002 | Screening | ₹5,25,000 | 1.00 | Yes |

Both tool calls returned successful MCP results with `is_error=False`.

## 18. SQLite Checkpointing

**Files:** `agent_workflow.py`, `checkpoint_demo.py`

LangGraph SQLite checkpointing is used to persist workflow state.

Database:

`checkpoints.sqlite`

The demonstration:

1. Executes the intent classification node.
2. Executes the RAG node.
3. Pauses before the final response node.
4. Saves the completed node results.
5. Resumes with the same thread ID.
6. Executes the remaining response node without rerunning completed nodes.

The checkpoint demonstration successfully resumed and passed response validation.

Run:

```bash
python checkpoint_demo.py
```

## 19. Timeouts and Retries

**File:** `reliability_demo.py`

Reliability demonstrations use LangGraph nodes and asynchronous execution.

### Retry policy

| Setting | Value |
|---|---:|
| Maximum attempts | 3 |
| Initial interval | 1 second |
| Backoff factor | 2 |
| Maximum interval | 4 seconds |
| Jitter | Enabled |

The simulated node fails on the first two calls and succeeds on the third call.

This demonstrates recovery from a transient failure using exponential backoff and jitter.

### Per-node timeout

The simulated slow node has a timeout of 1 second.

The demonstration raises a `NodeTimeoutError` instead of hanging.

### Global workflow timeout

The complete simulated workflow has a global timeout of 1.5 seconds.

The demonstration confirms that the whole workflow is cancelled when the execution time exceeds the configured limit.

Run:

```bash
python reliability_demo.py
```

---

# 20. Running the Project

## Step 1: Create and activate the virtual environment

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

## Step 2: Install dependencies

```bash
pip install -r requirements.txt
```

## Step 3: Generate the dataset

```bash
python dataset.py
```

## Step 4: Create the knowledge base

```bash
python create_knowledge_base.py
```

## Step 5: Build the vector database

```bash
python build_vector_db.py
```

This creates the local ChromaDB collections.

## Step 6: Run the main agent

```bash
python agent_workflow.py
```

## Step 7: Run the FastAPI backend

```bash
uvicorn api:api --reload
```

Open:

`http://127.0.0.1:8000/docs`

## Step 8: Run chunking evaluation

```bash
python evaluate_chunking.py
```

## Step 9: Run RAG evaluation

```bash
python rag_evaluation.py
```

## Step 10: Run the MCP server and client

In the first terminal:

```bash
python fastmcp_server.py
```

In a second terminal:

```bash
python fastmcp_client.py
```

## Step 11: Run checkpointing demonstration

```bash
python checkpoint_demo.py
```

## Step 12: Run reliability demonstrations

```bash
python reliability_demo.py
```

---

# 21. Offline and MOCK_LLM Execution

The project uses deterministic MOCK_LLM mode for answer generation and evaluation.

No external LLM API key is required.

The SentenceTransformer embedding model and ChromaDB operate locally after the embedding model has been downloaded and cached.

An internet connection may be required for the initial download of the embedding model and Python dependencies. After these are available locally, the graded demonstrations can run without external LLM API calls.

The graded demonstrations should use MOCK_LLM mode without external LLM API calls.

---

# 22. Conclusion

The Naukri.com Domain Support Agent combines recruitment knowledge retrieval, application-status lookup, LangGraph workflow orchestration, conversation memory, structured validation, guardrails, FastAPI deployment, structured logging, MCP integration and reliability demonstrations.

The project demonstrates the 16 tasks of the Masai School Recruitment & HR capstone through implementation and terminal-based test results.

**Project:** Naukri.com Domain Support Agent (LangGraph)  
**Track:** Recruitment & HR (Naukri.com)  
**Capstone:** Masai School