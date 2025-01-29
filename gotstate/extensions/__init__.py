"""
Extensions package for extension mechanisms.

Architecture:
- Defines extension interfaces
- Manages extension lifecycle
- Provides extension points
- Coordinates with sandbox
- Maintains extension boundaries

Design Patterns:
- Plugin Pattern: Extension loading
- Strategy Pattern: Extension behavior
- Observer Pattern: Extension events
- Proxy Pattern: Extension isolation
- Chain of Responsibility: Extension handling

Security:
- Extension isolation
- Resource control
- Access validation
- Sandbox enforcement
- Security boundaries

Cross-cutting:
- Error handling
- Performance monitoring
- Extension metrics
- Thread safety
"""

from .hooks import ExtensionHooks
from .sandbox import ExtensionSandbox

__all__ = ["ExtensionHooks", "ExtensionSandbox"]
