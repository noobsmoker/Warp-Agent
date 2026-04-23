"""
MCP (Model Context Protocol) Bridge
Allows external MCP servers to register as side agents and inject context.
"""

import asyncio
import json
from typing import Dict, Any, List, Optional, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib


class MCPMessageType(Enum):
    INITIALIZE = "initialize"
    TOOLS_LIST = "tools/list"
    TOOLS_CALL = "tools/call"
    PROMPTS_LIST = "prompts/list"
    PROMPTS_EXECUTE = "prompts/execute"
    RESOURCES_LIST = "resources/list"
    RESOURCES_READ = "resources/read"
    RESOURCE_SUBSCRIBE = "resource/subscribe"


@dataclass
class MCPServer:
    """Registered MCP server."""
    server_id: str
    name: str
    version: str
    capabilities: Dict[str, Any]
    tools: List[Dict[str, Any]] = field(default_factory=list)
    prompts: List[Dict[str, Any]] = field(default_factory=list)
    resources: List[Dict[str, Any]] = field(default_factory=list)
    connected_at: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    
    def is_alive(self, timeout_seconds: int = 60) -> bool:
        """Check if server is still connected."""
        age = (datetime.now() - self.last_seen).total_seconds()
        return age < timeout_seconds


@dataclass 
class MCPBridge:
    """
    Bridge for Model Context Protocol.
    Allows external servers to register and inject context into councils.
    """
    
    _servers: Dict[str, MCPServer] = field(default_factory=dict)
    _server_callbacks: Dict[str, Callable] = field(default_factory=dict)
    _context_injector: Optional[Callable] = field(default=None, init=False)
    
    def register_server(
        self,
        server_id: str,
        name: str,
        version: str,
        capabilities: Dict[str, Any]
    ) -> MCPServer:
        """
        Register an MCP server.
        
        Returns:
            Registered server info
        """
        server = MCPServer(
            server_id=server_id,
            name=name,
            version=version,
            capabilities=capabilities
        )
        
        self._servers[server_id] = server
        return server
    
    def unregister_server(self, server_id: str):
        """Unregister an MCP server."""
        if server_id in self._servers:
            del self._servers[server_id]
    
    def get_server(self, server_id: str) -> Optional[MCPServer]:
        """Get server info."""
        return self._servers.get(server_id)
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """List all registered servers."""
        return [
            {
                "server_id": s.server_id,
                "name": s.name,
                "version": s.version,
                "capabilities": s.capabilities,
                "tool_count": len(s.tools),
                "connected_at": s.connected_at.isoformat()
            }
            for s in self._servers.values()
            if s.is_alive()
        ]
    
    def set_context_injector(self, injector: Callable):
        """Set function to inject context into councils."""
        self._context_injector = injector
    
    def inject_context(
        self,
        server_id: str,
        context: str,
        priority: int = 0
    ):
        """
        Inject context from an MCP server into active councils.
        
        Args:
            server_id: Source server
            context: Context to inject
            priority: Higher = more important (affects injection order)
        """
        if self._context_injector:
            self._context_injector(server_id, context, priority)
    
    # === Server-Side Server Implementation ===
    
    async def handle_initialize(self, server_id: str, request: Dict) -> Dict:
        """Handle initialize request from a server."""
        params = request.get("params", {})
        
        server = MCPServer(
            server_id=server_id,
            name=params.get("protocolVersion", "unknown"),
            version=params.get("protocolVersion", "1.0"),
            capabilities=params.get("capabilities", {})
        )
        
        self._servers[server_id] = server
        
        return {
            "protocolVersion": "1.0",
            "capabilities": {
                "tools": {},
                "prompts": {},
                "resources": {}
            },
            "serverInfo": {
                "name": "warp-agent",
                "version": "0.1.0"
            }
        }
    
    async def handle_list_tools(self, server_id: str) -> List[Dict]:
        """List tools from a registered server."""
        server = self._servers.get(server_id)
        if not server:
            return []
        
        return server.tools
    
    async def handle_list_prompts(self, server_id: str) -> List[Dict]:
        """List prompts from a registered server."""
        server = self._servers.get(server_id)
        if not server:
            return []
        
        return server.prompts
    
    async def handle_list_resources(self, server_id: str) -> List[Dict]:
        """List resources from a registered server."""
        server = self._servers.get(server_id)
        if not server:
            return []
        
        return server.resources
    
    def update_server_tools(self, server_id: str, tools: List[Dict]):
        """Update a server's tool list."""
        if server_id in self._servers:
            self._servers[server_id].tools = tools
    
    def update_server_prompts(self, server_id: str, prompts: List[Dict]):
        """Update a server's prompt list."""
        if server_id in self._servers:
            self._servers[server_id].prompts = prompts
    
    def update_server_resources(self, server_id: str, resources: List[Dict]):
        """Update a server's resource list."""
        if server_id in self._servers:
            self._servers[server_id].resources = resources
    
    # === Council Integration ===
    
    async def delegate_to_council(
        self,
        server_id: str,
        task: str,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Delegate a task to a council on behalf of an MCP server.
        
        Returns:
            Council response
        """
        from core.agent_council import get_council
        
        # Determine which council type based on task
        if "code" in task.lower() or "execute" in task.lower():
            council_type = "code"
        elif "search" in task.lower() or "research" in task.lower():
            council_type = "research"
        elif "brainstorm" in task.lower() or "creative" in task.lower():
            council_type = "creative"
        else:
            council_type = "research"
        
        council = get_council()
        
        # Build prompt with MCP context
        mcp_prompt = f"[MCP DELEGATION from {server_id}]\nTask: {task}\nParams: {json.dumps(params)}\n\n"
        
        council_id = await council.spawn_council(
            prompt=mcp_prompt,
            council_types=[council_type]
        )
        
        # Wait for completion (in practice, would be async)
        responses = council.get_council_responses(council_id)
        
        return {
            "council_id": council_id,
            "council_type": council_type,
            "responses": responses
        }
    
    def get_capabilities(self) -> Dict[str, Any]:
        """Get bridge capabilities."""
        return {
            "server_registration": True,
            "context_injection": True,
            "council_delegation": True,
            "toolsync": True,
            "resource_sync": True
        }


# Global instance
_bridge: Optional[MCPBridge] = None


def get_mcp_bridge() -> MCPBridge:
    """Get or create the global MCP bridge."""
    global _bridge
    if _bridge is None:
        _bridge = MCPBridge()
    return _bridge