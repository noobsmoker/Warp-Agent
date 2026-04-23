# Warp-Claw: Warp-Cortex-Enabled Hermes Agent

**One-liner install:**

```bash
curl -fsSL https://raw.githubusercontent.com/noobsmoker/warp-claw/main/scripts/install.sh | bash
```

The installer works on Linux, macOS, WSL2, and Android (Termux). It automatically:

* Installs a Python 3.11 virtual environment
* Adds the required dependencies (`transformers`, `torch`, `gudhi`, plus all Hermes core deps)
* Sets up the `warp-claw` command in `~/.local/bin` (or `$PREFIX/bin` on Termux)
* Runs the Hermes setup wizard (you can skip it with `--no-wizard`)

After installation, start the agent with:

```bash
warp-claw               # interactive CLI (full Hermes UI)
warp-claw --tui         # Ink-based TUI
warp-claw gateway        # launch the messaging gateway (Telegram, Discord, etc.)
```

## Overview

Warp-Claw is the **Hermes Agent** enriched with the **Warp-Cortex** infrastructure. It provides:

* **Singleton weight sharing** – load a transformer model once and share it across all agents (O(1) weight memory).
* **Topological Synapse** – TDA-based landmark selection that preserves the context manifold while compressing the KV-cache (O(N·k) context memory).
* **KV-Cache sparsification** – witness-complex-inspired pruning for up to a **10×** reduction in memory usage.
* **Referential Injection** – asynchronous sub-agent updates without pausing generation.
* **Scalable multi-agent spawning** – up to 12+ concurrent sub-agents (configurable) via a shared-memory manager.

All of this is delivered in a **single binary-like install** that can be run on a cheap consumer GPU (e.g., RTX 4090) with **under 3 GB VRAM for 100 agents**.

## Features

| Feature | Description |
|---------|-------------|
| **One-Line Install** | `curl … | bash` – no manual dependencies. |
| **Unified CLI / TUI** | Classic prompt-toolkit CLI *or* modern Ink TUI (`warp-claw --tui`). |
| **Multi-Platform Gateways** | Telegram, Discord, Slack, WhatsApp, Signal, Matrix, etc. |
| **Modular Tool System** | 40+ built-in tools, plus community-contributed skills via the Skills Hub. |
| **Memory-Efficient Scaling** | Singleton model + Topological Synapse = constant weight memory, linear-in-k context memory. |
| **Referential Injection** | Sub-agents inject updates async, preserving streaming generation. |
| **Auto-Context Compression** | Built-in `compress` command reduces token usage while preserving information. |
| **Open-Source** | MIT license, fully auditable. |
| **Extensible Provider System** | New providers (e.g., `warp-cortex`) can be added via a simple transport class. |

## Warp-Cortex Architecture

Warp-Claw integrates four key components from the Warp-Cortex paper:

### 1. Singleton Weight Sharing
- **Purpose:** Eliminate duplicate model weights across agents
- **Implementation:** `WarpCortexSingleton` class loads transformer once
- **Memory Impact:** O(1) weight memory instead of O(N × L)

### 2. Topological Synapse
- **Purpose:** Compress context while preserving semantic structure
- **Implementation:** Gudhi TDA library for landmark selection
- **Memory Impact:** O(N × k) context memory, k ≪ L

### 3. KV-Cache Sparsification
- **Purpose:** Prune redundant attention patterns
- **Implementation:** Witness-complex-inspired pruning
- **Memory Impact:** Up to 10× reduction in KV-cache size

### 4. Referential Injection
- **Purpose:** Async sub-agent updates without stream disruption
- **Implementation:** asyncio queues and locks
- **Benefit:** Seamless parallel reasoning

### Performance Metrics

On a single RTX 4090:
- **100 concurrent agents:** 2.2 GB total VRAM
- **Theoretical scaling:** 1,000+ agents before compute bottleneck
- **Memory efficiency:** Linear scaling with agent count

## Installation

### Prerequisites

* `bash` (standard on Linux/macOS)
* `curl` and `git` (bundled on most systems)

### One-Line Install

```bash
curl -fsSL https://raw.githubusercontent.com/noobsmoker/warp-claw/main/scripts/install.sh | bash
```

The script will:

1. Detect the host platform (desktop/server vs. Termux).
2. Create a Python 3.11 virtual environment in `~/.warp-claw/venv`.
3. Install all runtime dependencies, including `transformers>=4.21`, `torch>=2.0`, `gudhi>=3.8`.
4. Symlink the `warp-claw` executable into `~/.local/bin` (or `$PREFIX/bin` on Android).
5. Optionally run `warp-claw setup` to finish configuration (you can skip with `--no-wizard`)

### Manual Installation

If you prefer manual setup:

```bash
git clone https://github.com/noobsmoker/warp-claw.git
cd warp-claw
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv venv --python 3.11
source venv/bin/activate
uv pip install -e ".[all]"
python -m warp_claw  # or symlink to ~/bin/warp-claw
```

## Configuration

Warp-Claw uses the same configuration system as Hermes Agent:

1. **User config:** `~/.warp-claw/config.yaml` (settings), `~/.warp-claw/.env` (API keys)
2. **Setup wizard:** `warp-claw setup` – configures providers, tools, and profiles
3. **Profiles:** Multiple isolated instances via `warp-claw -p <profile>`

### Key Configuration Options

```yaml
# ~/.warp-claw/config.yaml
model:
  default: "warp-cortex/gpt2"  # Use Warp-Cortex provider
  provider: "warp-cortex"

delegation:
  provider: "warp-cortex"      # Enable scalable sub-agent spawning
  max_concurrent_children: 12  # Scale beyond default 3

warp_cortex:
  model_name: "gpt2"           # Base model for singleton
  landmark_count: 64           # k parameter for synapse
  sparsification_threshold: 0.95  # Witness complex pruning
```

## Running the Agent

### Interactive CLI

```bash
warp-claw
```

Features:
- Multiline editing with syntax highlighting
- Slash commands (`/model`, `/tools`, `/skills`)
- Auto-completion and history
- Interrupt-and-redirect for long-running tasks

### Modern TUI

```bash
warp-claw --tui
```

React-based terminal UI with:
- Streaming responses
- Tool activity feeds
- Concurrent conversation tabs
- Keyboard shortcuts

### Messaging Gateway

```bash
warp-claw gateway
```

Connect via:
- Telegram: `/new`, `/model`, `/compress`
- Discord: Direct messages or channels
- Slack: Bot commands and threads
- WhatsApp: Via Twilio or similar
- Signal: Secure messaging

## Developer Guide

### Project Structure

```
warp-claw/
├── run_agent.py              # AIAgent class — core conversation loop
├── model_tools.py            # Tool orchestration
├── cli.py                    # HermesCLI class — interactive CLI
├── warp_cortex/              # Warp-Cortex implementation
│   ├── __init__.py           # Singleton model loader
│   ├── synapse.py            # Topological Synapse (TDA landmarks)
│   ├── sparsification.py     # KV-cache pruning
│   ├── injection.py          # Referential Injection
│   └── manager.py            # WarpCortexManager for scaling
├── tools/                    # Tool implementations
├── gateway/                  # Messaging platform adapters
├── ui-tui/                   # React TUI (Ink)
├── tests/                    # Pytest suite
└── scripts/install.sh        # One-line installer
```

### Adding Tools

1. Create `tools/your_tool.py` with `registry.register()`
2. Add to `toolsets.py` constants
3. Auto-discovery handles the rest

### Adding Providers

1. Create `agent/transports/your_provider.py`
2. Add to `hermes_cli/providers.py`
3. Implement `ProviderTransport` interface

### Testing

```bash
scripts/run_tests.sh                    # Full CI-parity suite
scripts/run_tests.sh tests/warp_cortex/ # Warp-Cortex specific tests
```

## Testing & Scalability

### Running the Scalability Test

```bash
python test_warp_cortex_scalability.py
```

This test:
- Spawns 12 concurrent agents
- Measures memory usage before/after
- Validates O(1) weight sharing
- Reports execution time and efficiency

### Expected Results

```
Memory usage before spawning agents: 150.2 MB
Memory usage after spawning agents: 850.1 MB
Total memory increase: 699.9 MB
Agents spawned: 12
Average memory per agent: 58.3 MB
Execution time: 2.34 seconds
```

### Benchmarking Larger Scales

For 100+ agents, modify the test script:

```python
# In test_warp_cortex_scalability.py
NUM_AGENTS = 100  # Scale up
```

Monitor with `nvidia-smi` for GPU memory usage.

## Licensing & Contributing

### License

MIT License - see LICENSE file.

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make changes with tests
4. Submit a pull request

### Code Standards

- **Python:** PEP 8 with Black formatting
- **Type hints:** Required for new code
- **Tests:** 100% coverage for new features
- **Documentation:** Update README for user-facing changes

### Community

- **Issues:** GitHub Issues for bugs/features
- **Discussions:** GitHub Discussions for questions
- **Discord:** Join the Warp-Claw community

## Credits

### Built on Hermes Agent

Warp-Claw is built upon **[Hermes Agent](https://github.com/nousresearch/hermes-agent)** — the self-improving AI agent framework by Nous Research. This project extends Hermes with Warp-Cortex scaling technology to enable massive multi-agent coordination.

**Hermes Agent provides:**
- Core AI agent architecture and conversation loops
- Comprehensive tool system (40+ tools)
- Multi-platform messaging gateways (Telegram, Discord, Slack, etc.)
- CLI/TUI interfaces and slash commands
- Skills system and memory management
- Open-source licensing (MIT)

**Warp-Claw adds:**
- Warp-Cortex scaling infrastructure
- Singleton weight sharing for memory efficiency
- Topological Synapse for context compression
- Million-agent coordination capabilities

### Warp-Cortex Research

Warp-Cortex technology is based on research by Jorge L. Ruiz Williams, published as "[Warp-Cortex: An Asynchronous, Memory-Efficient Architecture for Million-Agent Cognitive Scaling on Consumer Hardware](https://arxiv.org/abs/2601.01298)".

---

Built with ❤️ by the open-source AI community.
- Session management and memory
- Multi-channel messaging (Discord, Telegram, webchat, Signal, etc.)
- Tool orchestration and skill system
- Cron scheduling and heartbeats

### Additional Credits

- [Warp-Cortex](https://github.com/JorgeLRW/warp-cortex) - Multi-agent architecture inspiration
- [Qwen](https://github.com/QwenLM/Qwen) - Default models
- [Llama](https://github.com/meta-llama) - Alternative models
- [HuggingFace](https://huggingface.co) - Model hub

## License

MIT

---

⭐ Star us on GitHub if this helps!
