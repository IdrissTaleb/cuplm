"""Built-in skills and a helper to assemble the default tool registry."""

from __future__ import annotations

from cup.config import Config
from cup.skills.files import make_file_tools
from cup.skills.shell import make_shell_tool
from cup.tools import ToolRegistry

__all__ = ["build_default_registry", "make_file_tools", "make_shell_tool"]


def build_default_registry(config: Config) -> ToolRegistry:
    """The default toolset for local file & system automation."""
    registry = ToolRegistry()
    for tool in make_file_tools(config.workdir):
        registry.register(tool)
    if config.allow_shell:
        registry.register(make_shell_tool(config.workdir, config.shell_timeout))
    return registry
