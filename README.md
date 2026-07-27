# Synapse Agents

A **multi-agent AI system** that works fully offline using a local LLM (Ollama).
A team of specialised agents collaborates to plan, research, execute, and
self-review any high-level goal you give it.

```
User Goal ──► Planner ──► Researcher ──► Executor ──► Reflector ──► Final Output
                 │               │             │            │
                 └───────────────┴─────────────┴────────────┘
                              (per step, repeated)
```

---

## Architecture

```
/
├── agents/
│   ├── planner.py       # Breaks goal into ordered steps
│   ├── researcher.py    # Enriches each step with context
│   ├── executor.py      # Produces concrete implementations
│   └── reflector.py     # Reviews & improves executor output
├── core/
│   ├── llm.py           # Reusable Ollama API wrapper
│   ├── memory.py        # SQLite task history & knowledge base
│   └── orchestrator.py  # State-machine pipeline driver
├── api/
│   └── routes.py        # FastAPI endpoints
├── utils/
│   └── helpers.py       # Logging, file I/O, safe shell, plugin loader
├── frontend/
│   └── index.html       # Single-file React-free UI (served by FastAPI)
├── tests/
│   └── test_agents.py   # 76 unit tests (all offline, fully mocked)
├── main.py              # Entry point (API server or CLI)
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| LLM | [Ollama](https://ollama.com) (mistral / llama3) |
| API | FastAPI + Uvicorn |
| Memory | SQLite (built-in) |
| Validation | Pydantic v2 |
| Testing | pytest |

---

## Prerequisites

1. **Python 3.10+**
2. **[Ollama](https://ollama.com/download)** installed and running
3. A pulled model, e.g. `mistral`:

```bash
ollama pull mistral
```

---

## Setup & Installation

```bash
# 1. Clone the repo
git clone https://github.com/Mirdula18/Synapse-Agents.git
cd Synapse-Agents

# 2. Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Start Ollama (in a separate terminal)
ollama serve
```

---

## Running the API Server

```bash
python main.py
# Server starts at http://localhost:8000
```

- **Swagger UI**: http://localhost:8000/docs
- **Frontend UI**: http://localhost:8000/
- **Health check**: http://localhost:8000/health

Runtime tuning via environment variables:

```bash
# Optional auth (if set, requests must include header X-API-Key)
set SYNAPSE_API_KEY=your-secret-key

# Default model used by API/UI when not overridden in request
set SYNAPSE_DEFAULT_MODEL=mistral

# Faster failure / retry behavior for slow models
set OLLAMA_CONNECT_TIMEOUT_S=10
set OLLAMA_READ_TIMEOUT_S=45
set OLLAMA_RETRIES=2
set OLLAMA_RETRY_BACKOFF_S=1.5

# Response-size and temperature tuning for speed/stability
set OLLAMA_NUM_PREDICT=700
set OLLAMA_TEMPERATURE=0.2
```

Custom options:

```bash
python main.py --host 0.0.0.0 --port 8080 --model llama3 --log-level DEBUG
```

---

## CLI Mode (no HTTP server)

```bash
# Run a task directly
python main.py --cli --goal "Build me a portfolio website"

# Use llama3 instead of mistral
python main.py --cli --goal "Set up a FastAPI project" --model llama3

# Interactive mode – approve plan before execution
python main.py --cli --interactive --goal "Create a REST API"

# Disable self-reflection for faster runs
python main.py --cli --no-reflect --goal "Write a sorting algorithm"
```

---

## API Endpoints

API mode is always non-interactive. For human approval before execution,
use CLI mode with `--interactive`.

### Authentication (optional)

If `SYNAPSE_API_KEY` is set, include `X-API-Key` in every API request.

```bash
curl -X POST http://localhost:8000/run-task \
  -H "Content-Type: application/json" \
  -H "X-API-Key: your-secret-key" \
  -d '{"goal":"Build a portfolio website","model":"mistral","enable_reflection":true}'
```

Frontend UI note:
- The built-in UI includes an API key input and sends `X-API-Key` automatically when provided.

### `POST /run-task`

Execute a goal through the full pipeline.

```json
// Request
{
  "goal": "Build a portfolio website",
  "model": "mistral",
  "enable_reflection": true
}

// Response
{
  "task_id": 1,
  "goal": "Build a portfolio website",
  "status": "completed",
  "plan": {
    "goal": "Build a portfolio website",
    "steps": ["...", "..."],
    "estimated_complexity": "medium"
  },
  "final_output": {
    "goal": "Build a portfolio website",
    "total_steps": 5,
    "completed_steps": 5,
    "step_outputs": [...]
  },
  "elapsed_seconds": 42.1
}
```

### `POST /run-task-async`

Submit a background job and return immediately with a `job_id`.

```json
{
  "goal": "Build a portfolio website",
  "model": "mistral",
  "enable_reflection": true
}
```

Response:

```json
{ "job_id": "...", "status": "queued", "goal": "Build a portfolio website" }
```

### `GET /run-task-async/{job_id}`

Poll job state, progress events, and final result.

### `GET /history`

Returns the 20 most recent tasks (configurable via `?limit=N`).

### `GET /task/{id}`

Returns full details for a task including all step results.

### `GET /health`

```json
{ "status": "ok", "ollama_available": true, "model": "mistral" }
```

### `GET /models`

Returns installed Ollama models for UI dropdown selection.

```json
{ "models": ["mistral", "llama3"], "default_model": "mistral" }
```

---

## Agents

### Planner Agent
- Receives the user goal
- Returns `{ goal, steps[], estimated_complexity, confidence }`

### Researcher Agent
- Enriches each step with details, resources, best practices and pitfalls
- Searches the local knowledge base for relevant prior knowledge

### Executor Agent
- Produces concrete code, configurations, or answers for each step
- Returns `{ step, result, code, explanation, status, confidence }`

### Reflector Agent (Self-Reflection)
- Reviews the executor's output
- Assigns a `quality_score` (0–1)
- Automatically improves the output when quality is below threshold

---

## Memory System

All tasks and step results are persisted in a local SQLite database
(`data/synapse_memory.db`).  The knowledge base is searched automatically
by the Researcher agent so previous task knowledge is reused.

```python
from core.memory import list_tasks, get_task, search_knowledge

tasks = list_tasks(limit=10)
task  = get_task(task_id=1)
hints = search_knowledge("FastAPI routing")
```

---

## Features

| Feature | Status |
|---|---|
| Offline operation (Ollama) | ✅ |
| Planner agent | ✅ |
| Researcher agent | ✅ |
| Executor agent | ✅ |
| Self-reflection agent | ✅ |
| Confidence scoring | ✅ |
| SQLite task history | ✅ |
| Knowledge base (reusable memory) | ✅ |
| Error handling with retries | ✅ |
| Interactive plan approval (CLI mode) | ✅ |
| FastAPI REST API | ✅ |
| Web UI (single-file, no build step) | ✅ |
| Safe shell execution | ✅ |
| File reader / writer utilities | ✅ |
| Plugin system (dynamic agent loading) | ✅ |
| Progress callback / streaming | ✅ |
| Full unit test suite (76 tests) | ✅ |

---


## Plugin System

Add a new agent without modifying the core:

```python
# plugins/my_agent.py
class MyAgent:
    def run(self, step: str, **kwargs) -> dict:
        ...

# Load dynamically
from utils.helpers import load_agent_plugin
MyAgent = load_agent_plugin("plugins.my_agent.MyAgent")
```

---

## CORS Configuration (Brief)

- CORS is environment-driven.
- Credentials are enabled by default (`SYNAPSE_CORS_ALLOW_CREDENTIALS=true`).
- When credentials are enabled, wildcard origin `*` is rejected at startup.
- In development (`SYNAPSE_ENV=development`), safe localhost origins are allowed by default.
- In production (`SYNAPSE_ENV=production`), set explicit `SYNAPSE_CORS_ORIGINS`.

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `SYNAPSE_API_KEY` | unset | If set, API requires `X-API-Key` header |
| `SYNAPSE_DEFAULT_MODEL` | `mistral` | Default model for API/UI requests |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama server URL |
| `SYNAPSE_ENV` | `development` | Runtime mode (`development` or `production`) |
| `SYNAPSE_CORS_ALLOW_CREDENTIALS` | `true` | Enables credentialed CORS |
| `SYNAPSE_CORS_ORIGINS` | localhost defaults in dev | Comma-separated allowed origins |
| `SYNAPSE_HISTORY_LIMIT` | `20` | Default `/history` limit |
| `OLLAMA_CONNECT_TIMEOUT_S` | `10` | Ollama connect timeout (seconds) |
| `OLLAMA_READ_TIMEOUT_S` | `60` | Ollama read timeout (seconds) |
| `OLLAMA_RETRIES` | `1` | LLM request retry attempts |
| `OLLAMA_RETRY_BACKOFF_S` | `1.5` | Initial retry backoff (seconds) |
| `OLLAMA_NUM_PREDICT` | `520` | Max token budget per generation |
| `OLLAMA_TEMPERATURE` | `0.2` | LLM sampling temperature |

---

## License

MIT
