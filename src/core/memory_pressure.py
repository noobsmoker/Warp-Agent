"""
M1 Memory Governor - Memory Pressure Handling for Unified Memory

Addresses:
- Memory exhaustion causing kernel panics
- Graceful degradation under pressure
- Agent count limits based on available memory
"""

import psutil
import asyncio
import gc
import torch
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datetime import datetime


@dataclass
class MemoryThresholds:
    """Memory pressure thresholds."""
    warning: float = 0.75  # 75% of total memory
    critical: float = 0.90  # 90% - start killing agents
    emergency: float = 0.95  # 95% - hard stop


@dataclass
class AgentMemoryInfo:
    """Tracks memory usage per agent."""
    agent_id: str
    priority: int  # 0 = highest, 10 = lowest
    memory_mb: float
    last_activity: datetime = field(default_factory=datetime.now)


class M1MemoryGovernor:
    """
    Controls agent count based on available unified memory.
    M1 doesn't have dedicated VRAM - shared with CPU.
    """
    
    def __init__(self, max_agents: int = 100):
        self.max_agents = max_agents
        self.thresholds = MemoryThresholds()
        self.active_agents: Dict[str, AgentMemoryInfo] = {}
        self.agent_semaphore = asyncio.Semaphore(max_agents)
        self.memory_history: List[float] = []
        self._monitoring = False
        self._callbacks: List[callable] = []
        
    async def start_monitoring(self):
        """Start continuous memory monitoring."""
        self._monitoring = True
        while self._monitoring:
            await self._check_memory()
            await asyncio.sleep(1)
    
    def stop_monitoring(self):
        """Stop memory monitoring."""
        self._monitoring = False
    
    async def _check_memory(self):
        """Check memory and take action if needed."""
        mem = psutil.virtual_memory()
        pressure = mem.percent / 100
        
        self.memory_history.append(pressure)
        if len(self.memory_history) > 60:  # Keep last 60s
            self.memory_history.pop(0)
        
        if pressure > self.thresholds.emergency:
            await self._emergency_purge()
        elif pressure > self.thresholds.critical:
            await self._graceful_degradation()
        elif pressure > self.thresholds.warning:
            self._reduce_batch_sizes()
        
        # Notify callbacks
        for callback in self._callbacks:
            try:
                callback(pressure, len(self.active_agents))
            except:
                pass
    
    async def _emergency_purge(self):
        """Emergency - reduce to minimum agents."""
        print(f"[MEMORY] Emergency: {self.thresholds.emergency*100}% usage - hard purge")
        
        # Keep only highest priority agents
        sorted_agents = sorted(
            self.active_agents.values(),
            key=lambda x: x.priority
        )
        
        keep_count = max(5, len(sorted_agents) // 4)  # Keep 25% or minimum 5
        to_remove = sorted_agents[keep_count:]
        
        for agent in to_remove:
            await self._remove_agent(agent.agent_id)
        
        await self._force_gc()
    
    async def _graceful_degradation(self):
        """Reduce agent count when memory pressure builds."""
        current = len(self.active_agents)
        target = int(current * 0.6)  # Reduce to 60%
        
        print(f"[MEMORY] Critical: {self.thresholds.critical*100}% - reducing from {current} to {target}")
        
        # Cancel lowest priority agents
        sorted_agents = sorted(
            self.active_agents.values(),
            key=lambda x: x.priority
        )
        
        to_remove = sorted_agents[target:]
        for agent in to_remove:
            await self._remove_agent(agent.agent_id)
        
        await self._force_gc()
    
    def _reduce_batch_sizes(self):
        """Reduce batch sizes when approaching warning."""
        # This would be used by batch processing to reduce sizes
        pass
    
    async def _remove_agent(self, agent_id: str):
        """Remove an agent from tracking."""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
    
    async def _force_gc(self):
        """Force garbage collection and clear MPS cache."""
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    
    def register_agent(self, agent_id: str, priority: int = 5, estimated_memory_mb: float = 50):
        """Register a new agent."""
        self.active_agents[agent_id] = AgentMemoryInfo(
            agent_id=agent_id,
            priority=priority,
            memory_mb=estimated_memory_mb
        )
    
    def unregister_agent(self, agent_id: str):
        """Unregister an agent."""
        if agent_id in self.active_agents:
            del self.active_agents[agent_id]
    
    def get_safe_agent_count(self, requested: int) -> int:
        """Calculate safe agent count given current memory."""
        mem = psutil.virtual_memory()
        available_gb = (mem.total - mem.used) / (1024**3)
        
        # Conservative: 50MB per agent on M1
        max_safe = int(available_gb * 1024 / 50)
        
        return min(requested, max_safe, self.max_agents)
    
    def add_callback(self, callback: callable):
        """Add a callback for memory pressure changes."""
        self._callbacks.append(callback)
    
    def get_stats(self) -> Dict:
        """Get current memory statistics."""
        mem = psutil.virtual_memory()
        return {
            "total_gb": round(mem.total / (1024**3), 2),
            "used_gb": round(mem.used / (1024**3), 2),
            "available_gb": round(mem.available / (1024**3), 2),
            "percent": mem.percent,
            "active_agents": len(self.active_agents),
            "max_agents": self.max_agents,
            "history_avg": sum(self.memory_history) / len(self.memory_history) if self.memory_history else 0
        }