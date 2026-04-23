"""
File System Tool
File operations for Warp-Claw with permission checks.
"""

import os
import asyncio
import aiofiles
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .base_tool import BaseTool, ToolResult, ToolRegistry


class FileSystemTool(BaseTool):
    """
    File system operations: read, write, list, delete.
    """
    
    name = "file_system"
    description = "Read, write, and manage files in allowed directories."
    category = "file"
    tags = ["file", "filesystem", "io"]
    default_timeout = 10
    
    # Allowed base directories (security)
    ALLOWED_DIRS = [
        os.path.expanduser("~/warp-agent"),
        "/tmp/warp-agent",
    ]
    
    def __init__(self, allowed_dirs: Optional[List[str]] = None):
        super().__init__()
        if allowed_dirs:
            self.ALLOWED_DIRS.extend(allowed_dirs)
        self._ensure_dirs()
    
    def _ensure_dirs(self):
        """Ensure allowed directories exist."""
        for d in self.ALLOWED_DIRS:
            os.makedirs(d, exist_ok=True)
    
    def _check_path(self, path: str) -> Optional[str]:
        """Check if path is allowed."""
        abs_path = os.path.abspath(path)
        
        for allowed in self.ALLOWED_DIRS:
            if abs_path.startswith(os.path.abspath(allowed)):
                return None
        
        return f"Path not in allowed directories: {self.ALLOWED_DIRS}"
    
    async def execute(
        self,
        operation: str,
        path: str,
        content: Optional[str] = None,
        **kwargs
    ) -> ToolResult:
        """
        Execute file operation.
        
        Args:
            operation: read, write, list, delete, exists
            path: File path
            content: Content for write operation
            
        Returns:
            ToolResult with operation result
        """
        # Security check
        error = self._check_path(path)
        if error:
            return ToolResult(success=False, result=None, error=error)
        
        if operation == "read":
            return await self._read(path)
        elif operation == "write":
            return await self._write(path, content or "")
        elif operation == "list":
            return await self._list(path)
        elif operation == "delete":
            return await self._delete(path)
        elif operation == "exists":
            return self._exists(path)
        else:
            return ToolResult(
                success=False,
                result=None,
                error=f"Unknown operation: {operation}"
            )
    
    async def _read(self, path: str) -> ToolResult:
        """Read file contents."""
        try:
            async with aiofiles.open(path, 'r') as f:
                content = await f.read()
            
            return ToolResult(
                success=True,
                result={
                    "path": path,
                    "content": content,
                    "size": len(content)
                },
                tokens_estimate=len(content) // 4
            )
        except FileNotFoundError:
            return ToolResult(success=False, result=None, error="File not found")
        except Exception as e:
            return ToolResult(success=False, result=None, error=str(e))
    
    async def _write(self, path: str, content: str) -> ToolResult:
        """Write file contents."""
        try:
            # Ensure parent exists
            parent = os.path.dirname(path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            
            async with aiofiles.open(path, 'w') as f:
                await f.write(content)
            
            return ToolResult(
                success=True,
                result={
                    "path": path,
                    "written": len(content),
                    "size": len(content)
                },
                tokens_estimate=len(content) // 4
            )
        except Exception as e:
            return ToolResult(success=False, result=None, error=str(e))
    
    async def _list(self, path: str) -> ToolResult:
        """List directory contents."""
        try:
            if not os.path.exists(path):
                return ToolResult(success=False, result=None, error="Directory not found")
            
            if not os.path.isdir(path):
                return ToolResult(success=False, result=None, error="Not a directory")
            
            entries = []
            for entry in os.listdir(path):
                full_path = os.path.join(path, entry)
                stat = os.stat(full_path)
                
                entries.append({
                    "name": entry,
                    "type": "dir" if os.path.isdir(full_path) else "file",
                    "size": stat.st_size if os.path.isfile(full_path) else 0,
                    "modified": stat.st_mtime
                })
            
            return ToolResult(
                success=True,
                result={
                    "path": path,
                    "entries": entries,
                    "count": len(entries)
                },
                tokens_estimate=len(entries) * 10
            )
        except Exception as e:
            return ToolResult(success=False, result=None, error=str(e))
    
    async def _delete(self, path: str) -> ToolResult:
        """Delete file or directory."""
        try:
            if not os.path.exists(path):
                return ToolResult(success=False, result=None, error="Path not found")
            
            if os.path.isdir(path):
                import shutil
                shutil.rmtree(path)
            else:
                os.unlink(path)
            
            return ToolResult(
                success=True,
                result={
                    "path": path,
                    "deleted": True
                }
            )
        except Exception as e:
            return ToolResult(success=False, result=None, error=str(e))
    
    def _exists(self, path: str) -> ToolResult:
        """Check if path exists."""
        return ToolResult(
            success=True,
            result={
                "path": path,
                "exists": os.path.exists(path),
                "is_file": os.path.isfile(path) if os.path.exists(path) else False,
                "is_dir": os.path.isdir(path) if os.path.exists(path) else False
            }
        )
    
    def get_schema(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "list", "delete", "exists"],
                        "description": "File operation"
                    },
                    "path": {
                        "type": "string",
                        "description": "File path"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content for write operation"
                    }
                },
                "required": ["operation", "path"]
            }
        }


# Register tool
ToolRegistry.register(FileSystemTool())