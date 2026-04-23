"""
OpenClaw MCP Client Integration for Warp Agent
Enables warp-agent agents to call OpenClaw tools via Model Context Protocol
"""

import asyncio
import json
import os
import logging
from typing import Dict, List, Any, Optional
from contextlib import AsyncExitStack
from dataclasses import dataclass

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# MCP imports
try:
    from mcp import ClientSession
    from mcp.client.stdio import stdio_client
    from mcp.client.streamable_http import streamablehttp_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logger.warning("MCP not installed. Run: pip install 'mcp[cli]'")


@dataclass
class OpenClawToolResult:
    """Result from calling an OpenClaw tool"""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None


class OpenClawMCPClient:
    """
    MCP Client that connects to OpenClaw's tool server
    Allows warp-agent agents to use OpenClaw's tools (web search, code execution, etc.)
    """

    def __init__(self, openclaw_url: str = None):
        """
        Initialize the OpenClaw MCP client

        Args:
            openclaw_url: URL to OpenClaw's MCP endpoint 
            (default: env var OPENCLAW_URL or localhost:3000/mcp)
        """
        self.openclaw_url = openclaw_url or os.getenv("OPENCLAW_URL", "http://localhost:3000/mcp")
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._available_tools: List[Dict[str, Any]] = []
        """
        Initialize the OpenClaw MCP client

        Args:
            openclaw_url: URL to OpenClaw's MCP endpoint 
            (default: localhost:3000/mcp for local OpenClaw)
        """
        self.openclaw_url = openclaw_url
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self._available_tools: List[Dict[str, Any]] = []

    async def connect(self) -> bool:
        """
        Connect to OpenClaw's MCP server

        Returns:
            True if connection successful, False otherwise
        """
        if not MCP_AVAILABLE:
            print("✗ MCP not installed. Run: pip install 'mcp[cli]'")
            return False

        try:
            # Use streamable HTTP transport to connect to OpenClaw
            transport = await self.exit_stack.enter_async_context(
                streamablehttp_client(self.openclaw_url)
            )
            read, write, get_session_id = transport

            # Create client session
            self.session = await self.exit_stack.enter_async_context(
                ClientSession(read, write)
            )

            # Initialize the session
            await self.session.initialize()

            # Fetch available tools from OpenClaw
            await self._refresh_tools()

            print(f"✓ Connected to OpenClaw MCP at {self.openclaw_url}")
            print(f"✓ Available tools: {[t.get('name') for t in self._available_tools]}")

            return True

        except Exception as e:
            logger.error(f"✗ Failed to connect to OpenClaw: {e}")
            return False

    async def _refresh_tools(self):
        """Refresh the list of available tools from OpenClaw"""
        if not self.session:
            return

        response = await self.session.list_tools()
        self._available_tools = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.inputSchema
            }
            for tool in response.tools
        ]

    def get_available_tools(self) -> List[Dict[str, Any]]:
        """Get list of available OpenClaw tools"""
        return self._available_tools

    async def call_tool(self, tool_name: str, arguments: Dict[str, Any]) -> OpenClawToolResult:
        """
        Call an OpenClaw tool via MCP

        Args:
            tool_name: Name of the tool to call (e.g., "web_search", "execute_python")
            arguments: Tool arguments as dictionary

        Returns:
            OpenClawToolResult with success status and result data
        """
        if not self.session:
            return OpenClawToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error="Not connected to OpenClaw"
            )

        try:
            # Call the tool through MCP
            result = await self.session.call_tool(tool_name, arguments)

            # Extract result content
            if result.content:
                # MCP returns content as a list of content objects
                text_results = []
                for content in result.content:
                    if hasattr(content, 'text'):
                        text_results.append(content.text)

                return OpenClawToolResult(
                    tool_name=tool_name,
                    success=True,
                    result="\n".join(text_results) if text_results else str(result.content),
                    error=None
                )
            else:
                return OpenClawToolResult(
                    tool_name=tool_name,
                    success=True,
                    result=None,
                    error=None
                )

        except Exception as e:
            return OpenClawToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e)
            )

    async def disconnect(self):
        """Clean up and disconnect from OpenClaw"""
        if self.exit_stack:
            await self.exit_stack.aclose()
        self.session = None
        print("✓ Disconnected from OpenClaw")


class WarpAgentOpenClawBridge:
    """
    Bridge that integrates OpenClaw tools into warp-agent's agent system
    This is the main interface used by warp-agent's council orchestrator
    """

    def __init__(self, openclaw_url: str = "http://localhost:3000/mcp"):
        self.mcp_client = OpenClawMCPClient(openclaw_url)
        self._connected = False

    async def initialize(self) -> bool:
        """Initialize the bridge and connect to OpenClaw"""
        self._connected = await self.mcp_client.connect()
        return self._connected

    def is_connected(self) -> bool:
        """Check if bridge is connected to OpenClaw"""
        return self._connected

    async def execute_tool(self, tool_name: str, **kwargs) -> str:
        """
        Execute an OpenClaw tool and return result as string
        This is the method called by warp-agent agents during their execution

        Args:
            tool_name: Tool to execute
            **kwargs: Tool arguments

        Returns:
            Tool result as string (for injection into agent context)
        """
        if not self._connected:
            return f"[Error: OpenClaw bridge not connected]"

        result = await self.mcp_client.call_tool(tool_name, kwargs)

        if result.success:
            return f"[Tool Result: {tool_name}]\n{result.result}"
        else:
            return f"[Tool Error: {tool_name}]\n{result.error}"

    async def get_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        Get tools formatted for LLM function calling
        Returns tools in OpenAI-compatible format
        """
        tools = self.mcp_client.get_available_tools()

        # Convert to OpenAI function calling format
        return [
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool["description"],
                    "parameters": tool["input_schema"]
                }
            }
            for tool in tools
        ]

    async def shutdown(self):
        """Clean up resources"""
        await self.mcp_client.disconnect()
        self._connected = False