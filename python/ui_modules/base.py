from abc import ABC, abstractmethod
import streamlit as st

class DashboardModule(ABC):
    """
    Interface for Dashboard Modules (Strategy Pattern).
    Each module handles its own UI rendering and logic.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of the algorithm/module."""
        pass

    @abstractmethod
    def render(self, **kwargs):
        """Render the Streamlit UI for this module."""
        pass

class ModuleFactory:
    """
    Factory Pattern to manage and retrieve modules.
    """
    def __init__(self):
        self._modules = {}

    def register_module(self, module: DashboardModule):
        self._modules[module.name] = module

    def get_module(self, name: str) -> DashboardModule:
        return self._modules.get(name)

    def get_all_names(self):
        return list(self._modules.keys())
