"""DSPy integration for Wity, the typed-decision API."""

from .client import Wity
from .tools import wity_tools

__version__ = "0.2.0"

__all__ = ["Wity", "wity_tools"]
