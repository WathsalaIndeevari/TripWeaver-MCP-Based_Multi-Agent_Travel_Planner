"""Module for managing connections to Model Context Protocol (MCP) servers.

This module is the ONLY place within the agent code that connects to and interacts
with MCP servers. Other modules, such as nodes.py, must never import or interface
with the mcp_server package or servers directly.
"""

from __future__ import annotations

import os
import sys
from langchain_mcp_adapters.client import MultiServerMCPClient

# Dynamically construct the workspace root and inject it into the PYTHONPATH env
# variable so that the stdio subprocess can successfully import workspace modules.
_WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_env = os.environ.copy()
if "PYTHONPATH" in _env:
    _env["PYTHONPATH"] = _WORKSPACE_ROOT + os.pathsep + _env["PYTHONPATH"]
else:
    _env["PYTHONPATH"] = _WORKSPACE_ROOT

# Initialize MultiServerMCPClient configured with the tripweaver stdio server entry
client = MultiServerMCPClient({
    "tripweaver": {
        "transport": "stdio",
        "command": "python",
        "args": [os.path.join(_WORKSPACE_ROOT, "mcp_server", "server.py")],
        "env": _env,
    }
})


async def get_mcp_tools() -> list:
    """Fetch and return the list of LangChain-compatible tools from the MCP server."""
    return await client.get_tools()


def get_tool_by_name(tools: list, name: str):
    """Retrieve a specific tool from the provided tool list by its name.

    Args:
        tools: List of LangChain-compatible tools.
        name: Name of the tool to search for.

    Returns:
        The tool object matching the name, or None if not found.
    """
    return next((t for t in tools if t.name == name), None)
