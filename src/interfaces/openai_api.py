"""
OpenAI-Compatible API Server
Provides OpenAI-compatible HTTP endpoints for Warp-Claw.
"""

import uuid
import json
import asyncio
from typing import List, Optional, Literal, Dict, Any, Union
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel, Field
from contextlib import asynccontextmanager

# Import core modules
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.cortex_bridge import get_bridge, M1CortexBridge
from core.agent_council import get_council, AgentCouncil
from core.memory_synapse import get_synapse


app = FastAPI(
    title="Warp-Claw API",
    description="OpenAI-compatible API with multi-agent councils for Apple Silicon",
    version="0.1.0"
)


# === Request Models ===

class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    name: Optional[str] = None
    tool_call_id: Optional[str] = None


class ToolDefinition(BaseModel):
    type: Literal["function"] = "function"
    function: Dict[str, str]


class FunctionCall(BaseModel):
    name: str
    arguments: str


class ChatCompletionRequest(BaseModel):
    model: str = "qwen-0.5b"
    messages: List[ChatMessage]
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Literal["auto", "none"]] = "auto"
    stream: bool = False
    max_tokens: Optional[int] = Field(None, ge=1, le=4096)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(1.0, ge=0.0, le=1.0)


class CompletionRequest(BaseModel):
    model: str = "qwen-0.5b"
    prompt: str
    max_tokens: Optional[int] = Field(512, ge=1, le=4096)
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0)
    stream: bool = False


class SpawnCouncilRequest(BaseModel):
    prompt: str
    council_types: List[str]
    agent_count: Optional[Dict[str, int]] = None


# === Response Models ===

class ChatMessageResponse(BaseModel):
    role: str = "assistant"
    content: Optional[str] = None
    tool_calls: Optional[List[FunctionCall]] = None


class Choice(BaseModel):
    index: int
    message: ChatMessageResponse
    finish_reason: Literal["stop", "length", "tool_calls"]


class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[Choice]
    usage: Optional[Dict[str, int]] = None


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


# === Global State (Thread-Safe) ===
# Using ThreadSafeState for concurrent access
from src.core.critical_fixes import _model_state

# For backwards compatibility, provide dict-like access
class _ModelStateProxy:
    """Proxy for backwards compatibility with dict access."""
    @staticmethod
    def __getitem__(key):
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_model_state.get(key))
    
    @staticmethod
    def __setitem__(key, value):
        import asyncio
        loop = asyncio.get_event_loop()
        loop.run_until_complete(_model_state.set(key, value))
    
    @staticmethod
    def get(key, default=None):
        import asyncio
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(_model_state.get(key)) or default

initialized_models = _ModelStateProxy()


# === Endpoints ===

@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "Warp-Claw API",
        "version": "0.1.0",
        "description": "OpenAI-compatible API with multi-agent councils"
    }


@app.get("/v1/models")
async def list_models():
    """List available models."""
    import yaml
    from pathlib import Path
    
    config_path = Path(__file__).parent.parent / "config" / "models.yaml"
    models = []
    
    if config_path.exists():
        with open(config_path) as f:
            config = yaml.safe_load(f)
            model_configs = config.get("models", {})
            now = 1700000000
            
            for model_id in model_configs:
                models.append({
                    "id": model_id,
                    "object": "model",
                    "created": now,
                    "owned_by": "warp-agent"
                })
    
    # Add default
    if not models:
        models = [
            {"id": "qwen-0.5b", "object": "model", "created": 1700000000, "owned_by": "warp-agent"},
            {"id": "qwen-1.5b", "object": "model", "created": 1700000000, "owned_by": "warp-agent"},
        ]
    
    return {"object": "list", "data": models}


@app.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """Standard chat completion with optional council support."""
    bridge = get_bridge(request.model)
    council = get_council()
    synapse = get_synapse()
    
    # Convert messages to single prompt
    system_messages = [m.content for m in request.messages if m.role == "system"]
    user_messages = [m.content for m in request.messages if m.role == "user"]
    last_message = user_messages[-1] if user_messages else ""
    
    prompt = "\n".join(system_messages) + "\n" + last_message if system_messages else last_message
    
    # Detect councils to activate
    councils = bridge._detect_councils(prompt)
    
    # Handle streaming
    if request.stream:
        async def generate():
            response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
            created = int(uuid.uuid1().int % 1000000)
            
            # Stream header
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model, 'choices': [{'index': 0, 'delta': {'role': 'assistant'}, 'finish_reason': None}]})}\n\n"
            
            # Stream tokens
            async for chunk in bridge._generate_stream(prompt, request.max_tokens or 512, request.temperature or 0.7):
                yield chunk
            
            # Final chunk
            yield f"data: {json.dumps({'id': response_id, 'object': 'chat.completion.chunk', 'created': created, 'model': request.model, 'choices': [{'index': 0, 'delta': {}, 'finish_reason': 'stop'}]})}\n\n"
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no"
            }
        )
    
    # Standard generation with councils
    response = await bridge.generate(
        prompt=prompt,
        councils=councils,
        max_tokens=request.max_tokens or 512,
        temperature=request.temperature or 0.7
    )
    
    # Build response
    response_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(uuid.uuid1().int % 1000000)
    
    return {
        "id": response_id,
        "object": "chat.completion",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "message": {
                "role": "assistant",
                "content": response
            },
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": len(prompt.split()),
            "completion_tokens": len(response.split()),
            "total_tokens": len(prompt.split()) + len(response.split())
        }
    }


@app.post("/v1/completions")
async def completions(request: CompletionRequest):
    """Text completion endpoint."""
    bridge = get_bridge(request.model)
    
    if request.stream:
        async def generate():
            response_id = f"cmpl-{uuid.uuid4().hex[:8]}"
            created = int(uuid.uuid1().int % 1000000)
            
            yield f"data: {json.dumps({'id': response_id, 'object': 'text_completion', 'created': created, 'model': request.model})}\n\n"
            
            async for chunk in bridge._generate_stream(request.prompt, request.max_tokens or 512, request.temperature or 0.7):
                yield chunk
            
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(
            generate(),
            media_type="text/event-stream"
        )
    
    response = await bridge.generate(
        prompt=request.prompt,
        max_tokens=request.max_tokens or 512,
        temperature=request.temperature or 0.7
    )
    
    response_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(uuid.uuid1().int % 1000000)
    
    return {
        "id": response_id,
        "object": "text_completion",
        "created": created,
        "model": request.model,
        "choices": [{
            "index": 0,
            "text": response,
            "finish_reason": "stop"
        }]
    }


@app.post("/v1/agents/spawn")
async def spawn_council(request: SpawnCouncilRequest):
    """Explicitly spawn a council with specified agents."""
    bridge = get_bridge()
    council = get_council()
    synapse = get_synapse()
    
    # Set the model's generator on the council
    if bridge._model is not None:
        council.set_generator(lambda p, **kw: bridge._generate_main(p, kw.get("max_tokens", 256), kw.get("temperature", 0.7)))
    
    # Spawn the council
    council_id = await council.spawn_council(
        prompt=request.prompt,
        council_types=request.council_types,
        agent_counts=request.agent_count
    )
    
    # Get responses
    responses = council.get_council_responses(council_id)
    
    # Inject into synapse
    for r in responses:
        synapse.inject_consensus(council_id, r)
    
    return {
        "council_id": council_id,
        "status": "spawning",
        "council_types": request.council_types
    }


@app.get("/v1/agents/status/{council_id}")
async def get_council_status(council_id: str):
    """Get status of a spawned council."""
    council = get_council()
    status = council.get_council_status(council_id)
    
    if status.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Council not found")
    
    return status


@app.get("/v1/agents")
async def list_agents():
    """List all active agents."""
    council = get_council()
    councils = council.get_all_councils()
    
    return {
        "active_councils": councils,
        "council_count": len(councils)
    }


@app.delete("/v1/agents/{council_id}")
async def clear_council(council_id: str):
    """Clear a council from memory."""
    council = get_council()
    council.clear_council(council_id)
    
    return {"status": "cleared", "council_id": council_id}


# === Run Server ===

def run(host: str = "0.0.0.0", port: int = 8000):
    """Run the API server."""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run()