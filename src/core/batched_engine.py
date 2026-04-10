"""
Batched Cortex Engine - Efficient 100 Agent Generation

Addresses:
- Current implementation spawns 100 individual calls
- MPS performs poorly with small batch sizes
- Optimal batch size = 8 for MPS
"""

import torch
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor
import asyncio


@dataclass
class AgentTask:
    """Task configuration for an agent."""
    task_id: str
    task_type: str
    system_prompt: str
    priority: int = 5


@dataclass
class AgentResult:
    """Result from agent generation."""
    agent_id: str
    task_type: str
    thought: str
    status: str  # completed, failed, timeout
    latency_ms: float = 0
    tokens_generated: int = 0


class BatchedCortexEngine:
    """
    Batches 100 agent generations into efficient chunks.
    MPS optimal batch size = 8 (sweet spot for M1).
    """
    
    def __init__(
        self,
        model,
        tokenizer,
        optimal_batch_size: int = 8,
        max_tokens: int = 128,
        temperature: float = 0.7
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.optimal_batch_size = optimal_batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        
    async def generate_agents(
        self,
        tasks: List[AgentTask],
        context_prompt: str
    ) -> List[AgentResult]:
        """
        Execute 100+ agents in optimized batches.
        """
        if not tasks:
            return []
        
        # Split into micro-batches
        batches = [
            tasks[i:i + self.optimal_batch_size]
            for i in range(0, len(tasks), self.optimal_batch_size)
        ]
        
        all_results = []
        
        for batch_idx, batch in enumerate(batches):
            results = await self._process_batch(batch, context_prompt, batch_idx)
            all_results.extend(results)
            
            # Clear MPS cache between batches
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
        
        return all_results
    
    async def _process_batch(
        self,
        batch: List[AgentTask],
        context_prompt: str,
        batch_idx: int
    ) -> List[AgentResult]:
        """Process a single batch of tasks."""
        import time
        start = time.time()
        
        # Prepare batched prompts
        prompts = [
            f"{task.system_prompt}\n\nContext: {context_prompt}\n\nAnalyze:"
            for task in batch
        ]
        
        # Tokenize together (batched)
        try:
            inputs = self.tokenizer(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Ensure contiguous for MPS
            if not inputs['input_ids'].is_contiguous():
                inputs = {k: v.contiguous() for k, v in inputs.items()}
            
            # Single forward pass for entire batch
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    temperature=self.temperature,
                    use_cache=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode results
            results = []
            input_len = inputs['input_ids'].shape[1]
            
            for i, task in enumerate(batch):
                try:
                    output_tokens = outputs[i][input_len:]
                    text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                    
                    results.append(AgentResult(
                        agent_id=f"batch{batch_idx}_agent{i}",
                        task_type=task.task_type,
                        thought=text,
                        status="completed",
                        latency_ms=(time.time() - start) * 1000,
                        tokens_generated=len(output_tokens)
                    ))
                except Exception as e:
                    results.append(AgentResult(
                        agent_id=f"batch{batch_idx}_agent{i}",
                        task_type=task.task_type,
                        thought="",
                        status=f"failed: {str(e)}"
                    ))
                    
        except Exception as e:
            # Return failed results for entire batch
            return [
                AgentResult(
                    agent_id=f"batch{batch_idx}_agent{i}",
                    task_type=task.task_type,
                    thought="",
                    status=f"failed: {str(e)}"
                )
                for i, task in enumerate(batch)
            ]
        
        return results
    
    async def generate_single(
        self,
        task: AgentTask,
        context_prompt: str
    ) -> AgentResult:
        """Generate a single agent (for compatibility)."""
        results = await self.generate_agents([task], context_prompt)
        return results[0] if results else AgentResult(
            agent_id=task.task_id,
            task_type=task.task_type,
            thought="",
            status="failed: no result"
        )


class AgentCouncil:
    """
    Manages a council of agents with voting/consensus.
    Uses batched generation for efficiency.
    """
    
    def __init__(self, engine: BatchedCortexEngine, num_agents: int = 7):
        self.engine = engine
        self.num_agents = num_agents
    
    async def run_council(
        self,
        task_description: str,
        system_prompt: str = "You are a wise council member."
    ) -> Dict:
        """Run a council decision with multiple agents."""
        tasks = [
            AgentTask(
                task_id=f"council_member_{i}",
                task_type="council",
                system_prompt=system_prompt,
                priority=5
            )
            for i in range(self.num_agents)
        ]
        
        results = await self.engine.generate_agents(tasks, task_description)
        
        # Simple consensus - return all thoughts
        return {
            "votes": [r.thought for r in results if r.status == "completed"],
            "num_responses": len([r for r in results if r.status == "completed"]),
            "all_results": results
        }