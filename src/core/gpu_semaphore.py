"""
MPS GPU Semaphore - Controls concurrent MPS operations

Addresses:
- MPS doesn't support true async execution
- Concurrent operations cause crashes
- Need controlled concurrency (max 2 concurrent)
"""

import asyncio
import torch
import time
from typing import Callable, Any, List, Dict
from dataclasses import dataclass, field
from collections import deque


@dataclass
class OperationMetrics:
    """Metrics for GPU operations."""
    queued_operations: int = 0
    active_operations: int = 0
    completed_operations: int = 0
    failed_operations: int = 0
    total_execution_time: float = 0


class MPSSemaphore:
    """
    Controls concurrent MPS operations to prevent backend crashes.
    MPS performs poorly with high concurrency - max 2 concurrent.
    """
    
    def __init__(self, max_concurrent: int = 2):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.max_concurrent = max_concurrent
        self.metrics = OperationMetrics()
        self.operation_queue: deque = deque()
        self._running = False
        
    async def execute(self, operation: Callable, *args, **kwargs) -> Any:
        """Execute an operation with GPU semaphore."""
        async with self.semaphore:
            self.metrics.active_operations += 1
            
            # Synchronize before operation (MPS requirement)
            if torch.backends.mps.is_available():
                torch.mps.synchronize()
            
            start_time = time.time()
            
            try:
                if asyncio.iscoroutinefunction(operation):
                    result = await operation(*args, **kwargs)
                else:
                    result = await asyncio.to_thread(operation, *args, **kwargs)
                
                self.metrics.completed_operations += 1
                self.metrics.total_execution_time += time.time() - start_time
                
                return result
                
            except Exception as e:
                self.metrics.failed_operations += 1
                raise
                
            finally:
                self.metrics.active_operations -= 1
                # Synchronize after operation
                if torch.backends.mps.is_available():
                    torch.mps.synchronize()
    
    async def execute_many(
        self,
        operations: List[Callable],
        timeout: float = 300
    ) -> List[Any]:
        """
        Execute multiple operations with controlled concurrency.
        Uses semaphore to prevent MPS crashes.
        """
        results = []
        
        async def run_with_timeout(op):
            try:
                return await asyncio.wait_for(
                    self.execute(op),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                return {"error": "timeout"}
            except Exception as e:
                return {"error": str(e)}
        
        # Run with controlled concurrency
        tasks = [run_with_timeout(op) for op in operations]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return results
    
    async def batch_execute(
        self,
        operations: List[tuple]  # List of (func, args, kwargs)
    ) -> List[Any]:
        """
        Execute batch with controlled concurrency.
        Each item: (function, args, kwargs)
        """
        results = []
        
        for func, args, kwargs in operations:
            result = await self.execute(func, *args, **kwargs)
            results.append(result)
            
            # Small delay between operations for stability
            await asyncio.sleep(0.1)
        
        return results
    
    def get_metrics(self) -> Dict:
        """Get current operation metrics."""
        return {
            "queued": self.metrics.queued_operations,
            "active": self.metrics.active_operations,
            "completed": self.metrics.completed_operations,
            "failed": self.metrics.failed_operations,
            "avg_time": (
                self.metrics.total_execution_time / 
                max(self.metrics.completed_operations, 1)
            ),
            "max_concurrent": self.max_concurrent
        }
    
    def reset_metrics(self):
        """Reset operation metrics."""
        self.metrics = OperationMetrics()


class GPUScheduler:
    """
    High-level GPU task scheduler using MPSSemaphore.
    """
    
    def __init__(self, max_concurrent: int = 2):
        self.semaphore = MPSSemaphore(max_concurrent)
        self.queue: asyncio.Queue = asyncio.Queue()
        self._running = False
        
    async def start(self):
        """Start the scheduler."""
        self._running = True
        while self._running:
            try:
                task = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
                func, args, kwargs, future = task
                try:
                    result = await self.semaphore.execute(func, *args, **kwargs)
                    future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
            except asyncio.TimeoutError:
                continue
    
    def stop(self):
        """Stop the scheduler."""
        self._running = False
    
    async def submit(self, func: Callable, *args, **kwargs) -> Any:
        """Submit a task to the scheduler."""
        loop = asyncio.get_event_loop()
        future = loop.create_future()
        await self.queue.put((func, args, kwargs, future))
        return await future
    
    def get_stats(self) -> Dict:
        """Get scheduler stats."""
        return {
            "queue_size": self.queue.qsize(),
            "semaphore": self.semaphore.get_metrics()
        }