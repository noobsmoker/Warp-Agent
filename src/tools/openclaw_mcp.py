"""
OpenClaw MCP Client - Connect warp-agent to OpenClaw tools

Addresses: Option A (Recommended) - OpenClaw as Tool Provider
- warp-agent remains local inference engine
- MCP client connects to OpenClaw's tool server
- Agents call OpenClaw tools via MCP when needed
- No K8s - just network connectivity to OpenClaw instance
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
import sseclient

logger = logging.getLogger(__name__)


class ToolProvider(Enum):
    """Available tool providers."""
    OPENCLAW = "openclaw"
    LOCAL = "local"  # Fallback to local tools


@dataclass
class ToolCall:
    """Represents a tool call request."""
    name: str
    arguments: Dict[str, Any]
    call_id: str


@dataclass
class ToolResult:
    """Represents a tool call result."""
    call_id: str
    success: bool
    result: Any = None
    error: str = None


class OpenClawMCPClient:
    """
    MCP client that connects warp-agent to OpenClaw's tool server.
    
    Usage:
        client = OpenClawMCPClient("http://localhost:8080")
        await client.connect()
        
        # Call a tool
        result = await client.call_tool("web_search", {"query": "AI agents"})
        
        # Or use as context manager
        async with OpenClawMCPClient("http://localhost:8080") as client:
            result = await client.call_tool("read_file", {"path": "/project/main.py"})
    """
    
    def __init__(
        self,
        openclaw_url: str = "http://localhost:8080",
        api_key: str = None,
        timeout: int = 30,
        reconnect_attempts: int = 3
    ):
        self.openclaw_url = openclaw_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout
        self.reconnect_attempts = reconnect_attempts
        self._session: Optional[aiohttp.ClientSession] = None
        self._connected = False
        self._available_tools: List[Dict] = []
        
    async def __aenter__(self):
        await self.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.disconnect()
    
    async def connect(self) -> bool:
        """Connect to OpenClaw tool server."""
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        try:
            self._session = aiohttp.ClientSession(
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            )
            
            # Test connection with capabilities request
            async with self._session.get(
                f"{self.openclaw_url}/mcp/capabilities"
            ) as resp:
                if resp.status == 200:
                    capabilities = await resp.json()
                    self._available_tools = capabilities.get("tools", [])
                    self._connected = True
                    logger.info(f"Connected to OpenClaw with {len(self._available_tools)} tools")
                    return True
                else:
                    logger.error(f"OpenClaw connection failed: {resp.status}")
                    return False
                    
        except Exception as e:
            logger.error(f"Failed to connect to OpenClaw: {e}")
            return False
    
    async def disconnect(self):
        """Disconnect from OpenClaw tool server."""
        if self._session:
            await self._session.close()
            self._session = None
        self._connected = False
    
    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> ToolResult:
        """
        Call a tool on OpenClaw.
        
        Example:
            result = await client.call_tool("web_search", {"query": "LLM agents"})
            if result.success:
                print(result.result)
        """
        if not self._connected:
            return ToolResult(
                call_id="",
                success=False,
                error="Not connected to OpenClaw"
            )
        
        call_id = f"call_{asyncio.get_event_loop().time()}"
        
        try:
            async with self._session.post(
                f"{self.openclaw_url}/mcp/tools/{tool_name}",
                json=arguments
            ) as resp:
                if resp.status == 200:
                    result = await resp.json()
                    return ToolResult(
                        call_id=call_id,
                        success=True,
                        result=result
                    )
                else:
                    error_text = await resp.text()
                    return ToolResult(
                        call_id=call_id,
                        success=False,
                        error=f"HTTP {resp.status}: {error_text}"
                    )
                    
        except asyncio.TimeoutError:
            return ToolResult(
                call_id=call_id,
                success=False,
                error="Request timeout"
            )
        except Exception as e:
            return ToolResult(
                call_id=call_id,
                success=False,
                error=str(e)
            )
    
    async def call_tools_batch(self, calls: List[ToolCall]) -> List[ToolResult]:
        """Call multiple tools in batch."""
        results = []
        
        # Process sequentially to avoid overwhelming the server
        for call in calls:
            result = await self.call_tool(call.name, call.arguments)
            result.call_id = call.call_id
            results.append(result)
        
        return results
    
    async def list_tools(self) -> List[Dict]:
        """List available tools from OpenClaw."""
        if not self._connected:
            return []
        
        try:
            async with self._session.get(
                f"{self.openclaw_url}/mcp/tools"
            ) as resp:
                if resp.status == 200:
                    return await resp.json()
                return []
        except:
            return []
    
    def is_connected(self) -> bool:
        """Check if connected to OpenClaw."""
        return self._connected
    
    def get_available_tools(self) -> List[str]:
        """Get list of available tool names."""
        return [t.get("name") for t in self._available_tools]


class OpenClawToolExecutor:
    """
    Executes OpenClaw tools with retry logic and fallback.
    
    Use this in warp-agent agents when they need external tools.
    """
    
    def __init__(
        self,
        openclaw_url: str = "http://localhost:8080",
        api_key: str = None,
        local_fallback: bool = True
    ):
        self.client = OpenClawMCPClient(openclaw_url, api_key)
        self.local_fallback = local_fallback
        self._local_tools: Dict[str, Callable] = {}
        
    async def __aenter__(self):
        await self.client.connect()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.disconnect()
    
    def register_local_tool(self, name: str, func: Callable):
        """Register a local fallback tool."""
        self._local_tools[name] = func
    
    async def execute(self, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Execute a tool, falling back to local if OpenClaw unavailable.
        """
        # Try OpenClaw first
        if self.client.is_connected():
            result = await self.client.call_tool(tool_name, arguments)
            if result.success:
                return result.result
        
        # Fallback to local tool if enabled
        if self.local_fallback and tool_name in self._local_tools:
            logger.info(f"Falling back to local tool: {tool_name}")
            return await self._local_tools[tool_name](arguments)
        
        raise RuntimeError(f"Tool {tool_name} unavailable (OpenClaw disconnected, no local fallback)")


# Convenience functions for common OpenClaw tools

async def openclaw_web_search(query: str, openclaw_url: str = "http://localhost:8080") -> List[Dict]:
    """Quick web search via OpenClaw."""
    async with OpenClawMCPClient(openclaw_url) as client:
        result = await client.call_tool("web_search", {"query": query})
        if result.success:
            return result.result
        raise RuntimeError(f"Web search failed: {result.error}")


async def openclaw_read_file(path: str, openclaw_url: str = "http://localhost:8080") -> str:
    """Quick file read via OpenClaw."""
    async with OpenClawMCPClient(openclaw_url) as client:
        result = await client.call_tool("read_file", {"path": path})
        if result.success:
            return result.result
        raise RuntimeError(f"File read failed: {result.error}")


async def openclaw_browse(url: str, openclaw_url: str = "http://localhost:8080") -> str:
    """Quick URL browse via OpenClaw."""
    async with OpenClawMCPClient(openclaw_url) as client:
        result = await client.call_tool("web_fetch", {"url": url})
        if result.success:
            return result.result
        raise RuntimeError(f"Browse failed: {result.error}")