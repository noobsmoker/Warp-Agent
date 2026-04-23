"""
Knowledge Graph Tool
Local RAG store using LanceDB for persistent memory.
"""

import os
import asyncio
import json
import hashlib
from typing import Dict, Any, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

from .base_tool import BaseTool, ToolResult, ToolRegistry


@dataclass
class KnowledgeEntry:
    """Single knowledge entry."""
    id: str
    content: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    created_at: str = ""
    updated_at: str = ""
    
    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.updated_at:
            self.updated_at = self.created_at


class KnowledgeGraph(BaseTool):
    """
    Local knowledge store for RAG.
    Supports add, search, and retrieve operations.
    """
    
    name = "knowledge_graph"
    description = "Store and retrieve knowledge with semantic search. Use for remembering facts, context, and long-term memory."
    category = "memory"
    tags = ["memory", "rag", "knowledge", "vector"]
    default_timeout = 30
    
    def __init__(self, data_dir: str = "/tmp/warp-agent/knowledge"):
        super().__init__()
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        self._entries: Dict[str, KnowledgeEntry] = {}
        self._index_file = os.path.join(data_dir, "index.json")
        self._load_index()
    
    def _load_index(self):
        """Load existing index."""
        if os.path.exists(self._index_file):
            try:
                with open(self._index_file) as f:
                    data = json.load(f)
                    for entry_data in data.get("entries", []):
                        entry = KnowledgeEntry(**entry_data)
                        self._entries[entry.id] = entry
            except Exception:
                pass
    
    def _save_index(self):
        """Save index to disk."""
        data = {
            "entries": [asdict(e) for e in self._entries.values()]
        }
        with open(self._index_file, 'w') as f:
            json.dump(data, f)
    
    def _generate_id(self, content: str) -> str:
        """Generate ID from content hash."""
        return hashlib.md5(content.encode()).hexdigest()[:16]
    
    async def execute(
        self,
        operation: str,
        content: str = "",
        query: str = "",
        metadata: Optional[Dict[str, Any]] = None,
        limit: int = 5,
        **kwargs
    ) -> ToolResult:
        """
        Execute knowledge operation.
        
        Args:
            operation: add, search, retrieve, list
            content: Content to add
            query: Query for search
            metadata: Metadata for entry
            limit: Maximum results
            
        Returns:
            ToolResult with results
        """
        if operation == "add":
            return await self._add(content, metadata or {})
        elif operation == "search":
            return await self._search(query, limit)
        elif operation == "retrieve":
            return self._retrieve(content)
        elif operation == "list":
            return self._list(limit)
        elif operation == "delete":
            return await self._delete(content)
        else:
            return ToolResult(
                success=False,
                result=None,
                error=f"Unknown operation: {operation}"
            )
    
    async def _add(self, content: str, metadata: Dict[str, Any]) -> ToolResult:
        """Add knowledge entry."""
        entry_id = self._generate_id(content)
        
        entry = KnowledgeEntry(
            id=entry_id,
            content=content,
            metadata=metadata
        )
        
        self._entries[entry_id] = entry
        self._save_index()
        
        return ToolResult(
            success=True,
            result={
                "id": entry_id,
                "added": True,
                "total_entries": len(self._entries)
            },
            tokens_estimate=len(content) // 4
        )
    
    async def _search(self, query: str, limit: int) -> ToolResult:
        """Search knowledge (simple text match for now)."""
        query_lower = query.lower()
        results = []
        
        # Simple keyword matching (replace with embeddings for production)
        for entry in self._entries.values():
            score = 0
            content_lower = entry.content.lower()
            
            # Count keyword matches
            for word in query_lower.split():
                if word in content_lower:
                    score += content_lower.count(word)
            
            if score > 0:
                results.append({
                    "id": entry.id,
                    "content": entry.content[:200],
                    "score": score,
                    "metadata": entry.metadata
                })
        
        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)
        
        return ToolResult(
            success=True,
            result={
                "query": query,
                "results": results[:limit],
                "count": len(results[:limit])
            },
            tokens_estimate=sum(len(r["content"]) for r in results[:limit]) // 4
        )
    
    def _retrieve(self, entry_id: str) -> ToolResult:
        """Retrieve specific entry."""
        entry = self._entries.get(entry_id)
        
        if not entry:
            return ToolResult(
                success=False,
                result=None,
                error=f"Entry not found: {entry_id}"
            )
        
        return ToolResult(
            success=True,
            result={
                "id": entry.id,
                "content": entry.content,
                "metadata": entry.metadata,
                "created_at": entry.created_at,
                "updated_at": entry.updated_at
            },
            tokens_estimate=len(entry.content) // 4
        )
    
    def _list(self, limit: int) -> ToolResult:
        """List all entries."""
        entries = [
            {
                "id": e.id,
                "content": e.content[:100],
                "metadata": e.metadata,
                "created_at": e.created_at
            }
            for e in list(self._entries.values())[:limit]
        ]
        
        return ToolResult(
            success=True,
            result={
                "entries": entries,
                "count": len(entries),
                "total": len(self._entries)
            }
        )
    
    async def _delete(self, entry_id: str) -> ToolResult:
        """Delete an entry."""
        if entry_id not in self._entries:
            return ToolResult(
                success=False,
                result=None,
                error=f"Entry not found: {entry_id}"
            )
        
        del self._entries[entry_id]
        self._save_index()
        
        return ToolResult(
            success=True,
            result={
                "id": entry_id,
                "deleted": True
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
                        "enum": ["add", "search", "retrieve", "list", "delete"],
                        "description": "Knowledge operation"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content for add/retrieve"
                    },
                    "query": {
                        "type": "string",
                        "description": "Query for search"
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Metadata for entry"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum results",
                        "default": 5
                    }
                },
                "required": ["operation"]
            }
        }


# Register tool
ToolRegistry.register(KnowledgeGraph())