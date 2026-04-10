"""
OpenClaw Tools Extension for OpenAI API
Adds /v1/tools endpoint to list available OpenClaw tools
"""

import asyncio
from typing import Dict, Any, List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

# Import the OpenClaw bridge
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.openclaw_mcp_client import get_bridge_with_openclaw, WarpClawOpenClawBridge


router = APIRouter(prefix="/v1", tags=["tools"])

# Global bridge instance
_openclaw_bridge: WarpClawOpenClawBridge = None


class ToolInfo(BaseModel):
    """Information about a tool"""
    name: str
    description: str
    parameters: Dict[str, Any]


class ToolsResponse(BaseModel):
    """Response with available tools"""
    tools: List[ToolInfo]
    source: str  # "openclaw" or "local"


@router.get("/tools", response_model=ToolsResponse)
async def list_tools():
    """
    List available tools from OpenClaw MCP server
    
    Returns:
        ToolsResponse with list of available tools
    """
    global _openclaw_bridge
    
    # Initialize bridge if needed
    if _openclaw_bridge is None:
        _openclaw_bridge = await get_bridge_with_openclaw(
            openclaw_url="http://localhost:3000/mcp"
        )
    
    # Get tools from OpenClaw
    if _openclaw_bridge and _openclaw_bridge.is_connected():
        openclaw_tools = await _openclaw_bridge.get_tools_for_llm()
        
        tools = []
        for t in openclaw_tools:
            func = t.get("function", {})
            tools.append(ToolInfo(
                name=func.get("name", ""),
                description=func.get("description", ""),
                parameters=func.get("parameters", {})
            ))
        
        return ToolsResponse(
            tools=tools,
            source="openclaw"
        )
    
    # Fallback to local tools
    return ToolsResponse(
        tools=[
            ToolInfo(
                name="web_search",
                description="Search the web for information",
                parameters={"type": "object", "properties": {"query": {"type": "string"}}}
            ),
            ToolInfo(
                name="execute_python",
                description="Execute Python code",
                parameters={"type": "object", "properties": {"code": {"type": "string"}}}
            ),
            ToolInfo(
                name="read_file",
                description="Read a file from disk",
                parameters={"type": "object", "properties": {"path": {"type": "string"}}}
            ),
        ],
        source="local"
    )


@router.post("/tools/execute")
async def execute_tool(tool_name: str, arguments: Dict[str, Any]):
    """
    Execute a tool directly
    
    Args:
        tool_name: Name of the tool to execute
        arguments: Tool arguments
    """
    global _openclaw_bridge
    
    if _openclaw_bridge is None:
        _openclaw_bridge = await get_bridge_with_openclaw()
    
    if _openclaw_bridge and _openclaw_bridge.is_connected():
        result = await _openclaw_bridge.execute_tool(tool_name, **arguments)
        return {"result": result}
    
    raise HTTPException(status_code=503, detail="OpenClaw not connected")


async def initialize_openclaw_tools(openclaw_url: str = "http://localhost:3000/mcp"):
    """Initialize OpenClaw connection on server startup"""
    global _openclaw_bridge
    
    try:
        _openclaw_bridge = await get_bridge_with_openclaw(openclaw_url=openclaw_url)
        print(f"✓ OpenClaw tools initialized: {len(_openclaw_bridge.get_available_tools()) if _openclaw_bridge.is_connected() else 0} tools")
    except Exception as e:
        print(f"⚠ OpenClaw tools not available: {e}")
        _openclaw_bridge = None


def setup_openclaw_routes(app):
    """Add OpenClaw tools routes to the FastAPI app"""
    app.include_router(router)
    
    # Add startup event to initialize OpenClaw
    @app.on_event("startup")
    async def startup_event():
        await initialize_openclaw_tools()