#!/usr/bin/env python3
"""
Warp Cortex Scalability Test Script

Tests spawning multiple subagents (15) using batched generation (Warp Cortex style)
vs individual generation (standard), measuring memory usage and performance.
"""

import time
import subprocess
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from src.core.batched_engine import BatchedCortexEngine, AgentTask
import asyncio
import os

def get_memory_usage():
    """Get current memory usage in MB."""
    result = subprocess.run(['ps', '-o', 'rss=', '-p', str(os.getpid())], capture_output=True, text=True)
    if result.returncode == 0:
        return int(result.stdout.strip()) / 1024
    return 0

def load_model():
    """Load a small model for testing."""
    # Use a small model for testing - you can change this
    model_name = "microsoft/DialoGPT-small"  # Small model for demo
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer

async def run_individual_agents(model, tokenizer, num_agents=15):
    """Run agents individually (standard approach)."""
    engine = BatchedCortexEngine(model, tokenizer, optimal_batch_size=1)

    tasks = [
        AgentTask(
            task_id=f"agent_{i}",
            task_type="analysis",
            system_prompt="You are an analyst.",
            priority=5
        )
        for i in range(num_agents)
    ]

    start_time = time.time()
    start_memory = get_memory_usage()

    results = []
    for task in tasks:
        result = await engine.generate_single(task, "Analyze this simple context.")
        results.append(result)

    end_time = time.time()
    end_memory = get_memory_usage()

    return {
        'time': end_time - start_time,
        'memory_delta': end_memory - start_memory,
        'results': results
    }

async def run_batched_agents(model, tokenizer, num_agents=15):
    """Run agents in batches (Warp Cortex approach)."""
    engine = BatchedCortexEngine(model, tokenizer, optimal_batch_size=8)

    tasks = [
        AgentTask(
            task_id=f"agent_{i}",
            task_type="analysis",
            system_prompt="You are an analyst.",
            priority=5
        )
        for i in range(num_agents)
    ]

    start_time = time.time()
    start_memory = get_memory_usage()

    results = await engine.generate_agents(tasks, "Analyze this simple context.")

    end_time = time.time()
    end_memory = get_memory_usage()

    return {
        'time': end_time - start_time,
        'memory_delta': end_memory - start_memory,
        'results': results
    }

async def main():
    print("Loading model...")
    model, tokenizer = load_model()

    print("Testing Warp Cortex Scalability with 15 agents...")

    print("\n1. Testing individual agent spawning (standard)...")
    individual_results = await run_individual_agents(model, tokenizer, 15)
    print(f"Time: {individual_results['time']:.2f}s")
    print(f"Memory delta: {individual_results['memory_delta']:.2f} MB")

    print("\n2. Testing batched agent spawning (Warp Cortex)...")
    batched_results = await run_batched_agents(model, tokenizer, 15)
    print(f"Time: {batched_results['time']:.2f}s")
    print(f"Memory delta: {batched_results['memory_delta']:.2f} MB")

    # Calculate efficiency
    time_efficiency = individual_results['time'] / batched_results['time']
    memory_efficiency = individual_results['memory_delta'] / batched_results['memory_delta']

    print("
Warp Cortex Efficiency Results:")
    print(f"Time efficiency: {time_efficiency:.2f}x faster")
    print(f"Memory efficiency: {memory_efficiency:.2f}x less memory")

    # Check completion rates
    individual_completed = len([r for r in individual_results['results'] if r.status == 'completed'])
    batched_completed = len([r for r in batched_results['results'] if r.status == 'completed'])

    print("
Completion Rates:")
    print(f"Individual: {individual_completed}/15")
    print(f"Batched: {batched_completed}/15")

if __name__ == "__main__":
    asyncio.run(main())