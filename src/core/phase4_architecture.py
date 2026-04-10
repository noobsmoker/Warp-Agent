"""
Phase 4: Architecture Improvements
- ARCH-001: Dependency Injection
- API-002/003/004/005: Missing AgentCouncil methods
- QUAL-004: Error handling for model loading
- DEVOPS-003: Health check endpoint
"""

import asyncio
import time
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum


# ============================================================
# ARCH-001: Dependency Injection Container
# ============================================================

class Container:
    """Simple dependency injection container."""
    
    _instance = None
    _services: Dict[str, Any] = {}
    _factories: Dict[str, callable] = {}
    
    @classmethod
    def get_instance(cls) -> 'Container':
        if cls._instance is None:
            cls._instance = Container()
        return cls._instance
    
    def register(self, name: str, instance: Any):
        """Register a singleton instance."""
        self._services[name] = instance
    
    def register_factory(self, name: str, factory: callable):
        """Register a factory function."""
        self._factories[name] = factory
    
    def resolve(self, name: str) -> Any:
        """Resolve a dependency."""
        if name in self._services:
            return self._services[name]
        if name in self._factories:
            return self._factories[name]()
        raise ValueError(f"Unknown service: {name}")
    
    def clear(self):
        """Clear all services."""
        self._services.clear()
        self._factories.clear()


# Convenience accessor
def get_container() -> Container:
    return Container.get_instance()


# ============================================================
# API Methods: Missing AgentCouncil Methods
# ============================================================

class AgentCouncilMethods:
    """Mixin class with missing API methods."""
    
    async def get_council_responses(self) -> Dict[str, List[str]]:
        """Get all responses from active councils."""
        if not hasattr(self, '_councils'):
            return {}
        
        return {
            council_id: list(responses) 
            for council_id, responses in self._councils.items()
        }
    
    async def get_council_status(self, council_id: str) -> Dict[str, Any]:
        """Get status of specific council."""
        if not hasattr(self, '_councils'):
            return {"status": "not_found"}
        
        if council_id not in self._councils:
            return {"status": "not_found"}
        
        return {
            "status": "active",
            "council_id": council_id,
            "agents": len(self._councils.get(council_id, [])),
            "type": getattr(self, '_council_types', {}).get(council_id, "unknown")
        }
    
    async def get_all_councils(self) -> List[Dict[str, Any]]:
        """Get all active councils."""
        if not hasattr(self, '_councils'):
            return []
        
        return [
            await self.get_council_status(council_id)
            for council_id in self._councils.keys()
        ]
    
    async def clear_council(self, council_id: str) -> bool:
        """Clear specific council."""
        if not hasattr(self, '_councils'):
            return False
        
        if council_id in self._councils:
            del self._councils[council_id]
            return True
        return False
    
    async def clear_all_councils(self) -> int:
        """Clear all councils."""
        if not hasattr(self, '_councils'):
            return 0
        
        count = len(self._councils)
        self._councils.clear()
        return count


# ============================================================
# QUAL-004: Error Handling for Model Loading
# ============================================================

class ModelLoadError(Exception):
    """Error loading model."""
    pass


class ModelLoader:
    """Model loader with comprehensive error handling."""
    
    ERROR_TYPES = {
        'NETWORK': 'Network error fetching model',
        'DISK_FULL': 'Not enough disk space',
        'PERMISSION': 'Permission denied',
        'CORRUPTED': 'Model files corrupted',
        'INCOMPATIBLE': 'Model incompatible with device',
        'MEMORY': 'Not enough memory',
        'UNKNOWN': 'Unknown error',
    }
    
    @classmethod
    def load_with_error_handling(cls, repo: str, device: str, dtype: torch.dtype):
        """Load model with proper error handling."""
        import psutil
        
        errors = []
        
        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.free < 5 * 1024 * 1024 * 1024:  # 5GB
            errors.append({
                'type': 'DISK_FULL',
                'message': f'Only {disk.free / (1024**3):.1f}GB free'
            })
        
        # Check memory
        mem = psutil.virtual_memory()
        if mem.available < 2 * 1024 * 1024 * 1024:  # 2GB
            errors.append({
                'type': 'MEMORY',
                'message': f'Only {mem.available / (1024**3):.1f}GB available'
            })
        
        if errors:
            raise ModelLoadError(errors)
        
        # Try loading
        try:
            from transformers import AutoModelForCausalLM
            model = AutoModelForCausalLM.from_pretrained(
                repo,
                device_map=device,
                torch_dtype=dtype,
                trust_remote_code=True
            )
            return model
        except ImportError as e:
            raise ModelLoadError([{'type': 'NETWORK', 'message': str(e)}])
        except Exception as e:
            raise ModelLoadError([{'type': 'UNKNOWN', 'message': str(e)}])


# ============================================================
# DEVOPS-003: Health Check Endpoint
# ============================================================

class HealthStatus(Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"


@dataclass
class HealthCheckResult:
    """Health check result."""
    status: HealthStatus
    checks: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)


class HealthChecker:
    """System health checker."""
    
    @classmethod
    async def check_all(cls) -> HealthCheckResult:
        """Run all health checks."""
        checks = {}
        all_healthy = True
        
        # Check model loading
        try:
            import torch
            checks['pytorch'] = {
                'status': 'ok',
                'mps_available': torch.backends.mps.is_available(),
                'cuda_available': torch.cuda.is_available()
            }
        except Exception as e:
            checks['pytorch'] = {'status': 'error', 'message': str(e)}
            all_healthy = False
        
        # Check memory
        try:
            import psutil
            mem = psutil.virtual_memory()
            mem_percent = mem.percent
            checks['memory'] = {
                'status': 'ok' if mem_percent < 90 else 'degraded',
                'percent': mem_percent
            }
            if mem_percent >= 90:
                all_healthy = False
        except Exception as e:
            checks['memory'] = {'status': 'error', 'message': str(e)}
        
        # Check disk
        try:
            disk = psutil.disk_usage('/')
            checks['disk'] = {
                'status': 'ok' if disk.percent < 90 else 'degraded',
                'percent': disk.percent
            }
            if disk.percent >= 90:
                all_healthy = False
        except Exception as e:
            checks['disk'] = {'status': 'error', 'message': str(e)}
        
        # Check model state
        try:
            from src.core.critical_fixes import _model_state
            model_loaded = await _model_state.get('current_model')
            checks['model'] = {
                'status': 'loaded' if model_loaded else 'not_loaded',
                'loaded': bool(model_loaded)
            }
        except Exception as e:
            checks['model'] = {'status': 'error', 'message': str(e)}
        
        status = HealthStatus.HEALTHY if all_healthy else HealthStatus.DEGRADED
        
        return HealthCheckResult(
            status=status,
            checks=checks
        )


# Health check endpoint for FastAPI
async def health_check_endpoint():
    """FastAPI health check endpoint."""
    result = await HealthChecker.check_all()
    
    return {
        "status": result.status.value,
        "checks": result.checks,
        "timestamp": result.timestamp
    }


print("✅ Phase 4 Architecture loaded:")
print("  - Container for dependency injection")
print("  - AgentCouncilMethods mixin")
print("  - ModelLoader with error handling")
print("  - HealthChecker for /health endpoint")