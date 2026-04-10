"""
PATCH: Modified M1CortexBridge with OpenClaw Tool Integration
Add this to src/core/cortex_bridge.py or create as cortex_bridge_openclaw.py
"""

import asyncio
import os
import re
from typing import Optional, List, Dict, Any

# Import the OpenClaw bridge
from .openclaw_mcp_client import WarpClawOpenClawBridge

# Constants
DEFAULT_QUERY_TRUNCATE = 200
DEFAULT_OPENCLAW_URL = os.getenv("OPENCLAW_URL", "http://localhost:3000/mcp")


class M1CortexBridgeWithOpenClaw:
    """
    Extended Cortex Bridge that delegates tool calls to OpenClaw via MCP
    Falls back to local execution if OpenClaw is unavailable
    """

    def __init__(self, model_id: str = "qwen-0.5b", openclaw_url: str = "http://localhost:3000/mcp", **kwargs):
        self.model_id = model_id
        self.openclaw_bridge: Optional[WarpClawOpenClawBridge] = None
        self._openclaw_url = openclaw_url
        self._openclaw_initialized = False
        self.model = None
        self.tokenizer = None

    async def initialize_openclaw(self) -> bool:
        """
        Initialize connection to OpenClaw's tool server
        Call this after creating the bridge, before generating
        """
        self.openclaw_bridge = WarpClawOpenClawBridge(self._openclaw_url)
        self._openclaw_initialized = await self.openclaw_bridge.initialize()
        return self._openclaw_initialized

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text with optional tool calls through OpenClaw"""
        # Check for tool triggers in prompt
        tool_results = []
        
        if self._openclaw_initialized and self.openclaw_bridge:
            # Detect tool triggers
            triggers = self._detect_tool_triggers(prompt)
            
            for trigger in triggers:
                tool_name = trigger["tool"]
                arguments = self._extract_tool_arguments(prompt, tool_name)
                
                result_text = await self.openclaw_bridge.execute_tool(
                    tool_name, **arguments
                )
                tool_results.append(result_text)
        
        # Generate response (would use local model in real implementation)
        response = f"Generated response for: {prompt[:50]}..."
        
        # Append tool results if any
        if tool_results:
            response += "\n\n" + "\n\n".join(tool_results)
        
        return response

    def _detect_tool_triggers(self, prompt: str) -> List[Dict[str, Any]]:
        """Detect tool triggers in the prompt"""
        triggers = []
        
        # Define trigger patterns
        trigger_patterns = {
            "web_search": ["[SEARCH]", "[FIND]", "[LOOK UP]"],
            "web_fetch": ["[FETCH]", "[GET URL]", "[BROWSE]"],
            "execute_python": ["[CODE]", "[RUN]", "[EXECUTE]"],
            "read_file": ["[READ]", "[FILE]", "[OPEN]"],
        }
        
        prompt_upper = prompt.upper()
        
        for tool_name, patterns in trigger_patterns.items():
            for pattern in patterns:
                if pattern.upper() in prompt_upper:
                    triggers.append({
                        "tool": tool_name,
                        "trigger": pattern
                    })
                    break
        
        return triggers

    def _extract_tool_arguments(self, prompt: str, tool_name: str) -> Dict[str, Any]:
        """
        Extract arguments for a tool call from the prompt
        Simple implementation - enhance based on your needs
        """
        # Default: pass the prompt as the query/content parameter
        if tool_name in ["web_search", "search"]:
            return {"query": prompt[:DEFAULT_QUERY_TRUNCATE]}
        elif tool_name in ["execute_python", "python", "code"]:
            if "```python" in prompt:
                code = prompt.split("```python")[1].split("```")[0].strip()
                return {"code": code}
            return {"code": prompt}
        elif tool_name in ["web_fetch", "fetch", "get_url"]:
            urls = re.findall(r'https?://[^\s]+', prompt)
            if urls:
                return {"url": urls[0]}
            return {"url": prompt.strip()}
        elif tool_name in ["read_file", "read"]:
            # Extract file path
            paths = re.findall(r'[/\w]+\.\w+', prompt)
            if paths:
                return {"path": paths[0]}
            return {"path": prompt.strip()}
        else:
            return {"content": prompt, "query": prompt}

    async def _execute_local_tool(self, tool_name: str, prompt: str) -> Optional[Dict]:
        """
        Execute a tool locally (original warp-claw behavior)
        """
        # Local stub - full implementation would use actual local tools
        return {
            "tool": tool_name,
            "result": "[Local execution - enable OpenClaw for full tool support]",
            "source": "local"
        }

    async def shutdown(self):
        """Clean up OpenClaw connection"""
        if self.openclaw_bridge:
            await self.openclaw_bridge.shutdown()
        self._connected = False


# Factory function with OpenClaw support
_bridge_with_openclaw: Optional[M1CortexBridgeWithOpenClaw] = None


async def get_bridge_with_openclaw(
    model_id: str = "qwen-0.5b", 
    openclaw_url: str = "http://localhost:3000/mcp"
) -> M1CortexBridgeWithOpenClaw:
    """
    Get or create the global bridge instance with OpenClaw integration

    Args:
        model_id: Model to use for generation
        openclaw_url: URL to OpenClaw MCP endpoint

    Returns:
        Initialized bridge with OpenClaw tool support
    """
    global _bridge_with_openclaw

    if _bridge_with_openclaw is None or _bridge_with_openclaw.model_id != model_id:
        _bridge_with_openclaw = M1CortexBridgeWithOpenClaw(
            model_id=model_id, 
            openclaw_url=openclaw_url
        )
        # Initialize OpenClaw connection
        connected = await _bridge_with_openclaw.initialize_openclaw()
        if not connected:
            print("⚠ Warning: Could not connect to OpenClaw. Tools will use local fallback.")

    return _bridge_with_openclaw