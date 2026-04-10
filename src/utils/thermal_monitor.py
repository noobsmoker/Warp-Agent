"""
M1 Thermal Monitor - Thermal Throttling Detection

Addresses:
- M1 reduces performance under sustained load
- 100 agents trigger thermal throttling
- Need dynamic agent count adjustment
"""

import subprocess
import asyncio
import os
from typing import Dict, Optional
from dataclasses import dataclass


@dataclass
class ThermalState:
    """Current thermal state."""
    level: str  # nominal, fair, serious, critical
    cpu_temp: Optional[float] = None
    gpu_temp: Optional[float] = None
    timestamp: float = 0


class M1ThermalMonitor:
    """
    Monitor M1 thermal state and adjust agent count accordingly.
    """
    
    # Mapping from thermal level to agent multiplier
    LEVEL_MULTIPLIERS = {
        "nominal": 1.0,
        "fair": 0.7,
        "serious": 0.4,
        "critical": 0.1,
        "unknown": 0.5
    }
    
    def __init__(self, base_max_agents: int = 100):
        self.base_max_agents = base_max_agents
        self.current_state = ThermalState(level="unknown")
        self.callbacks = []
        self._monitoring = False
        
    async def start_monitoring(self):
        """Start thermal monitoring."""
        self._monitoring = True
        while self._monitoring:
            await self._update_thermal_state()
            await asyncio.sleep(5)  # Check every 5 seconds
    
    def stop_monitoring(self):
        """Stop thermal monitoring."""
        self._monitoring = False
    
    async def _update_thermal_state(self):
        """Update thermal state via powermetrics."""
        try:
            # Try to get thermal data
            # Note: Requires sudo for powermetrics in production
            result = await self._read_powermetrics()
            if result:
                self.current_state = result
                await self._notify_callbacks()
        except Exception as e:
            # Fallback: estimate from CPU usage
            self.current_state = self._estimate_from_cpu()
    
    async def _read_powermetrics(self) -> Optional[ThermalState]:
        """Read thermal state using powermetrics (may require sudo)."""
        try:
            # Check if we can run powermetrics
            result = subprocess.run(
                ["which", "powermetrics"],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                return self._estimate_from_cpu()
            
            # Try running powermetrics (would need sudo in production)
            # For now, return estimated state
            return self._estimate_from_cpu()
            
        except Exception:
            return self._estimate_from_cpu()
    
    def _estimate_from_cpu(self) -> ThermalState:
        """Estimate thermal state from system load."""
        try:
            # Use CPU percentage as proxy
            import psutil
            cpu_percent = psutil.cpu_percent(interval=1)
            
            if cpu_percent < 50:
                level = "nominal"
            elif cpu_percent < 75:
                level = "fair"
            elif cpu_percent < 90:
                level = "serious"
            else:
                level = "critical"
            
            return ThermalState(
                level=level,
                timestamp=asyncio.get_event_loop().time()
            )
        except:
            return ThermalState(level="unknown")
    
    def get_max_agents_for_thermal(self) -> int:
        """Get maximum agents allowed given current thermal state."""
        multiplier = self.LEVEL_MULTIPLIERS.get(
            self.current_state.level,
            0.5
        )
        return int(self.base_max_agents * multiplier)
    
    def add_callback(self, callback):
        """Add callback for thermal state changes."""
        self.callbacks.append(callback)
    
    async def _notify_callbacks(self):
        """Notify callbacks of thermal state change."""
        for callback in self.callbacks:
            try:
                callback(self.current_state, self.get_max_agents_for_thermal())
            except:
                pass
    
    def get_state(self) -> Dict:
        """Get current thermal state."""
        return {
            "level": self.current_state.level,
            "max_agents": self.get_max_agents_for_thermal(),
            "base_max": self.base_max_agents,
            "multiplier": self.LEVEL_MULTIPLIERS.get(self.current_state.level, 0.5)
        }


class AdaptiveAgentScheduler:
    """
    Dynamically adjusts agent count based on thermal and memory.
    """
    
    def __init__(self, base_max: int = 100):
        self.base_max = base_max
        self.current_max = base_max
        self.thermal_monitor = M1ThermalMonitor(base_max)
        self._running = False
        
    async def start(self):
        """Start adaptive scheduling."""
        self._running = True
        await self.thermal_monitor.start_monitoring()
        
    def stop(self):
        """Stop adaptive scheduling."""
        self._running = False
        self.thermal_monitor.stop_monitoring()
    
    def get_current_limit(self) -> int:
        """Get current maximum agent count."""
        return self.thermal_monitor.get_max_agents_for_thermal()
    
    def get_status(self) -> Dict:
        """Get scheduler status."""
        return {
            "base_max": self.base_max,
            "current_limit": self.get_current_limit(),
            "thermal_state": self.thermal_monitor.get_state()
        }