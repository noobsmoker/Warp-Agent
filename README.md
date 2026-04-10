# 🤖 Warp-Claw

> Hybrid infrastructure combining Warp-Cortex with OpenAI-compatible API for local Apple Silicon deployment.

## Why Warp-Claw?

Warp-Claw bridges two powerful systems:
- **Warp-Cortex**: JorgeLRW's multi-agent architecture with River/Stream, Prism weight sharing, and Topological Synapse
- **OpenAI API**: Standard HTTP endpoints compatible with the entire OpenAI ecosystem

Deploy local AI agents with the same tooling you'd use with GPT-4.

## Features

| Feature | Description |
|---------|-------------|
| **Multi-Agent Councils** | Research, Code, Creative, and Meta councils that collaborate in real-time |
| **OpenAI-Compatible API** | Use `openai` Python client, curl, or any OpenAI-compatible tool |
| **M1/MPS Optimization** | Metal Performance Shaders for Apple Silicon |
| **Tool Integration** | Code execution, web search, file system, knowledge graph tools |
| **MCP Bridge** | Connect external Model Context Protocol servers |
| **WebSocket Streaming** | Real-time agent activity streaming |
| **Streamlit Dashboard** | Optional UI for monitoring and interaction |

## Architecture

```
User Request
     │
     ▼
┌────────────────────────────────────────┐
│   OpenAI API Server (:8000)           │
│   /v1/chat/completions                │
│   /v1/agents/spawn                   │
└─────────────────┬────────────────────┘
                  │
      ┌───────────┼───────────┐
      ▼           ▼           ▼
┌──────────┐ ┌────────┐ ┌──────────┐
│ Council  │ │ Tools  │ │WebSocket │
│Orchestr. │ │Execut. │ │ Stream   │
└────┬─────┘ └────┬───┘ └────┬────┘
     │            │
     ▼            ▼
┌─────────────────────────┐
│   M1 Cortex Bridge      │
│   (MPS/CPU)             │
└─────────────────────────┘
```

## Installation

```bash
# Clone the repo
git clone https://github.com/noobsmoker/warp-claw.git
cd warp-claw

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download a Model

```bash
python scripts/download_models.py qwen-0.5b

# Or use M1 setup script
bash scripts/setup_m1.sh
```

### 2. Run the API Server

```bash
python -m src.interfaces.openai_api
```

Server starts at `http://localhost:8000`

### 3. Use with OpenAI Client

```python
import openai

client = openai.OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

# Standard chat
response = client.chat.completions.create(
    model="qwen-0.5b",
    messages=[{"role": "user", "content": "Explain quantum computing [VERIFY]"}]
)
print(response.choices[0].message.content)
```

### 4. Spawn a Council

```python
import requests

r = requests.post("http://localhost:8000/v1/agents/spawn", json={
    "prompt": "Design a distributed system",
    "council_types": ["research", "creative", "code"],
    "agent_count": 10
})
council_id = r.json()["council_id"]

# Check status
r = requests.get(f"http://localhost:8000/v1/agents/status/{council_id}")
print(r.json())
```

### 5. Start Dashboard (Optional)

```bash
streamlit run src/dashboard/app.py
```

Dashboard at `http://localhost:8501`

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/v1/models` | GET | List available models |
| `/v1/chat/completions` | POST | Chat completion |
| `/v1/completions` | POST | Text completion |
| `/v1/agents/spawn` | POST | Spawn council |
| `/v1/agents/status/{id}` | GET | Council status |
| `/v1/agents` | GET | List agents |

## Configuration

### Models (`config/models.yaml`)

```yaml
models:
  qwen-0.5b:
    repo: "Qwen/Qwen2.5-0.5B-Instruct"
    device: "mps"
    max_agents: 100
    
  qwen-1.5b:
    repo: "Qwen/Qwen2.5-1.5B-Instruct"
    device: "mps"
    max_agents: 50

default_model: "qwen-0.5b"
```

### Agent Councils (`config/agents.yaml`)

```yaml
councils:
  research:
    agent_count: 3
    system_prompt: "You are a fact-checking sub-agent..."
    triggers: ["[SEARCH]", "[VERIFY]"]
    
  code:
    agent_count: 2
    system_prompt: "You are a code review sub-agent..."
    triggers: ["[CODE]", "[REVIEW]"]
```

## Tools

| Tool | Description |
|------|-------------|
| `execute_python` | Run Python code in sandbox |
| `web_search` | Search DuckDuckGo |
| `web_fetch` | Fetch URL content |
| `file_system` | Read/write files |
| `knowledge_graph` | RAG memory store |

## Makefile Commands

```bash
make help      # Show available targets
make setup    # Install & download models
make test     # Run tests
make run      # Start API server
make dashboard # Start dashboard
make clean   # Clean cache
```

## Docker

```bash
# Build
docker build -t warp-claw:latest .

# Run
docker run -p 8000:8000 -p 8501:8501 warp-claw:latest
```

## Requirements

- Python 3.11+
- Apple Silicon (M1/M2/M3) for MPS support
- Or x86_64 with CPU fallback

## Credits

### Built with OpenClaw
Warp-Claw was built using **[OpenClaw](https://openclaw.ai)** — the AI assistant framework that powers this project. OpenClaw provides:
- Session management and memory
- Multi-channel messaging (Discord, Telegram, webchat, Signal, etc.)
- Tool orchestration and skill system
- Cron scheduling and heartbeats

### Founders Profile
This project was generated by **OpenClaw** running with the founders profile — an autonomous AI agent configured to build, test, and deploy complete projects.

### Additional Credits

- [Warp-Cortex](https://github.com/JorgeLRW/warp-cortex) - Multi-agent architecture inspiration
- [Qwen](https://github.com/QwenLM/Qwen) - Default models
- [Llama](https://github.com/meta-llama) - Alternative models
- [HuggingFace](https://huggingface.co) - Model hub

## License

MIT

---

⭐ Star us on GitHub if this helps!