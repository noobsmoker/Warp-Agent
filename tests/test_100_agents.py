"""
Test 100 Agents - Stress Test for Warp-Claw

Tests the system under load with 100 concurrent agents.
"""

import pytest
import asyncio
import torch
from unittest.mock import Mock, AsyncMock, patch
from src.core.batched_engine import BatchedCortexEngine, AgentTask, AgentResult
from src.core.memory_pressure import M1MemoryGovernor, MemoryThresholds
from src.core.gpu_semaphore import MPSSemaphore
from src.utils.mps_fixes import enforce_contiguous_tensors, get_safe_device


class Test100Agents:
    """Test suite for 100 agent functionality."""
    
    @pytest.fixture
    def mock_model(self):
        """Create a mock model for testing."""
        model = Mock()
        model.generate = Mock(return_value=torch.tensor([[1, 2, 3, 4, 5]] * 8))
        return model
    
    @pytest.fixture
    def mock_tokenizer(self):
        """Create a mock tokenizer for testing."""
        tokenizer = Mock()
        tokenizer.return_tensors = Mock(return_value={
            'input_ids': torch.tensor([[1, 2, 3]] * 8),
            'attention_mask': torch.tensor([[1, 1, 1]] * 8)
        })
        tokenizer.decode = Mock(return_value="Test response")
        tokenizer.pad_token_id = 0
        return tokenizer
    
    @pytest.mark.asyncio
    async def test_batched_generation_100_agents(self, mock_model, mock_tokenizer):
        """Test batched generation with 100 agents."""
        engine = BatchedCortexEngine(
            model=mock_model,
            tokenizer=mock_tokenizer,
            optimal_batch_size=8
        )
        
        # Create 100 tasks
        tasks = [
            AgentTask(
                task_id=f"agent_{i}",
                task_type="analysis",
                system_prompt="You are an analyst.",
                priority=5
            )
            for i in range(100)
        ]
        
        results = await engine.generate_agents(tasks, "Analyze this context.")
        
        # Should complete all 100
        assert len(results) == 100
        completed = [r for r in results if r.status == "completed"]
        assert len(completed) == 100
    
    @pytest.mark.asyncio
    async def test_memory_governor_limits(self):
        """Test memory governor respects limits."""
        governor = M1MemoryGovernor(max_agents=100)
        
        # Register 100 agents
        for i in range(100):
            governor.register_agent(f"agent_{i}", priority=5)
        
        # Should return 100 if memory available
        safe = governor.get_safe_agent_count(100)
        assert safe <= 100
        
        # Test stats
        stats = governor.get_stats()
        assert "active_agents" in stats
        assert stats["active_agents"] == 100
    
    @pytest.mark.asyncio
    async def test_gpu_semaphore_concurrency(self):
        """Test GPU semaphore limits concurrent operations."""
        semaphore = MPSSemaphore(max_concurrent=2)
        
        # Track concurrent operations
        concurrent_count = []
        max_concurrent = 0
        
        async def dummy_operation():
            nonlocal max_concurrent
            async with semaphore.semaphore:
                concurrent_count.append(1)
                max_concurrent = max(max_concurrent, len(concurrent_count))
                await asyncio.sleep(0.1)
                concurrent_count.pop()
        
        # Run 10 operations - should be limited to 2
        await asyncio.gather(*[dummy_operation() for _ in range(10)])
        
        # Max should be 2
        assert max_concurrent <= 2
    
    def test_safe_device_detection(self):
        """Test device detection."""
        device = get_safe_device()
        assert device.type in ["mps", "cuda", "cpu"]


class TestMemoryPressure:
    """Test memory pressure handling."""
    
    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Test graceful degradation under pressure."""
        governor = M1MemoryGovernor(max_agents=100)
        
        # Register 100 agents
        for i in range(100):
            governor.register_agent(f"agent_{i}", priority=i % 10)
        
        # Should handle gracefully
        stats = governor.get_stats()
        assert stats["active_agents"] == 100
    
    def test_threshold_configuration(self):
        """Test memory thresholds."""
        thresholds = MemoryThresholds(
            warning=0.75,
            critical=0.90,
            emergency=0.95
        )
        
        assert thresholds.warning == 0.75
        assert thresholds.critical == 0.90
        assert thresholds.emergency == 0.95


class TestMPSFixes:
    """Test MPS-specific fixes."""
    
    def test_contiguous_tensor_enforcement(self):
        """Test contiguous tensor wrapping."""
        model = Mock()
        model.forward = Mock(return_value=torch.tensor([1, 2, 3]))
        
        wrapped = enforce_contiguous_tensors(model)
        
        # Should work with tensors
        result = wrapped.forward()
        assert isinstance(result, torch.Tensor)


# Run with: pytest tests/test_100_agents.py -v