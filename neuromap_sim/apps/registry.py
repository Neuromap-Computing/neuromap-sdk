"""Simple in-process plugin registry."""

from __future__ import annotations

from neuromap_sim.apps.base import BaseApp


class AppRegistry:
    """Registry of available applications."""

    def __init__(self) -> None:
        self._apps: dict[str, BaseApp] = {}

    def register(self, app: BaseApp) -> None:
        if app.name in self._apps:
            raise ValueError(f"An app named '{app.name}' is already registered.")
        self._apps[app.name] = app

    def get(self, name: str) -> BaseApp:
        try:
            return self._apps[name]
        except KeyError as exc:
            raise KeyError(f"Unknown app '{name}'. Available apps: {sorted(self._apps)}") from exc

    def list_apps(self) -> list[str]:
        return sorted(self._apps)
