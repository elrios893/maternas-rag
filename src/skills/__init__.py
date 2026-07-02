from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    required: list[str]
    fn: Callable


class ToolRegistry:
    _tools: ClassVar[dict[str, ToolSpec]] = {}

    @classmethod
    def register(cls, tool: ToolSpec) -> None:
        cls._tools[tool.name] = tool
        logger.info(f"[ToolRegistry] Tool registrada: '{tool.name}' — {tool.description}")

    @classmethod
    def execute(cls, name: str, **kwargs: Any) -> Any:
        tool = cls._tools.get(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' no encontrada. Tools disponibles: {list(cls._tools.keys())}")
        logger.info(f"[ToolRegistry] Ejecutando tool '{name}' con kwargs={kwargs}")
        return tool.fn(**kwargs)

    @classmethod
    def list_tools(cls) -> list[str]:
        return list(cls._tools.keys())


class Skill:
    name: str = ""
    description: str = ""
    tools: list[ToolSpec] = field(default_factory=list)

    def register_all(self) -> None:
        for tool in self.tools:
            ToolRegistry.register(tool)
        logger.info(f"[Skill] Skill '{self.name}' registrada con {len(self.tools)} tool(s)")
