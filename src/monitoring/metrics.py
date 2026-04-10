"""
Prometheus Metrics Integration

Addresses:
- Need metrics endpoint for monitoring
- Agent lifecycle tracking
- M1-specific metrics (memory, thermal)
- Batch performance tracking
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, generate_latest
from fastapi import APIRouter, Response
from typing import Dict, Optional
import time


class AgentMetrics:
    """Prometheus metrics for Warp-Claw agent system."""
    
    def __init__(self, port: int = 9090):
        self.port = port
        self.registry = CollectorRegistry()
        
        # ========== Agent Lifecycle ==========
        self.agents_spawned = Counter(
            'warpclaw_agents_spawned_total',
            'Total agents spawned',
            ['agent_type'],
            registry=self.registry
        )
        
        self.agents_completed = Counter(
            'warpclaw_agents_completed_total',
            'Total agents completed',
            ['status'],  # success, failed, timeout
            registry=self.registry
        )
        
        self.active_agents = Gauge(
            'warpclaw_active_agents',
            'Currently executing agents',
            registry=self.registry
        )
        
        # ========== Performance ==========
        self.agent_latency = Histogram(
            'warpclaw_agent_latency_seconds',
            'Agent generation latency',
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0, 60.0],
            registry=self.registry
        )
        
        self.batch_size = Histogram(
            'warpclaw_batch_size',
            'Number of agents per request',
            buckets=[1, 5, 10, 25, 50, 100],
            registry=self.registry
        )
        
        self.batch_latency = Histogram(
            'warpclaw_batch_latency_seconds',
            'Batch generation latency',
            buckets=[1, 5, 10, 30, 60, 120, 300],
            registry=self.registry
        )
        
        # ========== M1-Specific ==========
        self.memory_pressure = Gauge(
            'warpclaw_memory_pressure_ratio',
            'Current memory pressure (0-1)',
            registry=self.registry
        )
        
        self.thermal_level = Gauge(
            'warpclaw_thermal_level',
            'Thermal throttling level (0=nominal, 3=critical)',
            registry=self.registry
        )
        
        self.gpu_utilization = Gauge(
            'warpclaw_gpu_utilization_percent',
            'GPU utilization percentage',
            registry=self.registry
        )
        
        # ========== API ==========
        self.api_requests = Counter(
            'warpclaw_api_requests_total',
            'Total API requests',
            ['endpoint', 'method'],
            registry=self.registry
        )
        
        self.api_errors = Counter(
            'warpclaw_api_errors_total',
            'Total API errors',
            ['endpoint', 'error_type'],
            registry=self.registry
        )
        
        # ========== Token Usage ==========
        self.tokens_generated = Counter(
            'warpclaw_tokens_generated_total',
            'Total tokens generated',
            ['model'],
            registry=self.registry
        )
        
        # ========== Queue ==========
        self.queue_size = Gauge(
            'warpclaw_queue_size',
            'Current queue size',
            registry=self.registry
        )
        
    def record_spawn(self, agent_type: str):
        """Record agent spawn."""
        self.agents_spawned.labels(agent_type=agent_type).inc()
        self.active_agents.inc()
    
    def record_completion(self, status: str):
        """Record agent completion."""
        self.agents_completed.labels(status=status).inc()
        self.active_agents.dec()
    
    def record_latency(self, seconds: float):
        """Record agent latency."""
        self.agent_latency.observe(seconds)
    
    def record_batch(self, size: int, latency_seconds: float):
        """Record batch metrics."""
        self.batch_size.observe(size)
        self.batch_latency.observe(latency_seconds)
    
    def update_memory_pressure(self, ratio: float):
        """Update memory pressure."""
        self.memory_pressure.set(ratio)
    
    def update_thermal(self, level: int):
        """Update thermal level."""
        self.thermal_level.set(level)
    
    def update_gpu_util(self, percent: float):
        """Update GPU utilization."""
        self.gpu_utilization.set(percent)
    
    def record_api_request(self, endpoint: str, method: str):
        """Record API request."""
        self.api_requests.labels(endpoint=endpoint, method=method).inc()
    
    def record_api_error(self, endpoint: str, error_type: str):
        """Record API error."""
        self.api_errors.labels(endpoint=endpoint, error_type=error_type).inc()
    
    def record_tokens(self, model: str, count: int):
        """Record tokens generated."""
        self.tokens_generated.labels(model=model).inc(count)
    
    def update_queue(self, size: int):
        """Update queue size."""
        self.queue_size.set(size)
    
    def get_metrics(self) -> bytes:
        """Get Prometheus metrics output."""
        return generate_latest(self.registry)
    
    def get_stats(self) -> Dict:
        """Get metrics as dictionary for API response."""
        return {
            "active_agents": self.active_agents._value.get(),
            "memory_pressure": self.memory_pressure._value.get(),
            "thermal_level": self.thermal_level._value.get()
        }


# Global metrics instance
_metrics: Optional[AgentMetrics] = None


def get_metrics() -> AgentMetrics:
    """Get or create global metrics instance."""
    global _metrics
    if _metrics is None:
        _metrics = AgentMetrics()
    return _metrics


# FastAPI router for metrics endpoint
router = APIRouter(tags=["monitoring"])


@router.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(
        content=get_metrics().get_metrics(),
        media_type="text/plain"
    )


@router.get("/metrics/stats")
async def metrics_stats():
    """Get metrics stats as JSON."""
    return get_metrics().get_stats()


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": time.time()
    }