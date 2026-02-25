"""Application plugin contract."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any


class BaseApp(ABC):
    """Base class for app plugins."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique app name used by the registry."""

    @abstractmethod
    def run(self, config: Mapping[str, Any], **kwargs: Any) -> dict[str, Any]:
        """Execute the app and return serializable outputs."""
