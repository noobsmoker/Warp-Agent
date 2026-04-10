"""
Batch Agent Endpoint - Optimized 100 Agent Spawning

Addresses:
- Need /v1/agents/batch endpoint
- Return partial results if some agents timeout
- Include metrics
"""

from fastapi import APIRouter, BackgroundTasks, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional
from datetime import datetime


# Pydantic models
class AgentConfig(BaseModel):
    """Configuration for a single agent."""
    agent_id: str
    task_type: str
    system_prompt: str
    priority: int = 5


class BatchSpawnRequest(BaseModel):
    """Request for batch agent spawning."""
    prompt: str
    agent_configs: List[AgentConfig]
    max_tokens: int = 128
    temperature: float = 0.7
    return_partial: bool = True  # Return results even if some timeout


class AgentResult(BaseModel):
    """Result from a single agent."""
    agent_id: str
    task_type: str
    thought: str
    status: str  # completed, failed, timeout
    latency_ms: float = 0
    tokens_generated: int = 0


class BatchSpawnResponse(BaseModel):
    """Response from batch agent spawning."""
    completed: List[AgentResult]
    failed: List[AgentResult]
    partial: bool
    metrics: Dict


# Router
router = APIRouter(prefix="/v1/agents", tags=["agents"])


# In-memory engine reference (would be injected in production)
_engine = None


def set_engine(engine):
    """Set the batch engine (dependency injection)."""
    global _engine
    _engine = engine


@router.post("/batch", response_model=BatchSpawnResponse)
async def spawn_batch(
    request: BatchSpawnRequest,
    background_tasks: BackgroundTasks
):
    """
    Optimized endpoint for spawning 100 agents at once.
    
    Uses batched generation instead of individual spawns.
    Returns partial results if some agents timeout.
    """
    if not _engine:
        raise HTTPException(status_code=500, detail="Engine not initialized")
    
    if len(request.agent_configs) > 100:
        raise HTTPException(status_code=400, detail="Max 100 agents per batch")
    
    start_time = datetime.now()
    
    try:
        # Use batched generation
        results = await _engine.generate_agents(
            tasks=[
                {
                    "task_id": cfg.agent_id,
                    "task_type": cfg.task_type,
                    "system_prompt": cfg.system_prompt,
                    "priority": cfg.priority
                }
                for cfg in request.agent_configs
            ],
            context_prompt=request.prompt
        )
        
        completed = []
        failed = []
        
        for r in results:
            if r.status == "completed":
                completed.append(r)
            else:
                failed.append(r)
        
        elapsed = (datetime.now() - start_time).total_seconds()
        
        return BatchSpawnResponse(
            completed=completed,
            failed=failed,
            partial=len(failed) > 0 and request.return_partial,
            metrics={
                "total_agents": len(request.agent_configs),
                "success_rate": len(completed) / len(request.agent_configs) if request.agent_configs else 0,
                "elapsed_seconds": elapsed,
                "avg_latency_ms": (
                    sum(r.latency_ms for r in completed) / max(len(completed), 1)
                ),
                "batch_size": len(request.agent_configs)
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/stats")
async def get_agent_stats():
    """Get current agent statistics."""
    if not _engine:
        return {"status": "no_engine"}
    
    return {
        "engine": "batched",
        "max_batch_size": 100,
        "status": "ready"
    }


@router.post("/batch/async")
async def spawn_batch_async(
    request: BatchSpawnRequest,
    background_tasks: BackgroundTasks
):
    """
    Async version - returns immediately with job ID.
    Results delivered via webhook or polling.
    """
    import uuid
    
    job_id = str(uuid.uuid4())
    
    # Queue for background processing
    background_tasks.add_task(
        process_batch_async,
        job_id,
        request
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "estimated_completion": "30s"
    }


async def process_batch_async(job_id: str, request: BatchSpawnRequest):
    """Background task for async batch processing."""
    # Implementation would store results for polling
    pass