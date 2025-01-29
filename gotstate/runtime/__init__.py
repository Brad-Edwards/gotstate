"""
Runtime package for execution and monitoring.

Architecture:
- Manages event execution and run-to-completion
- Handles time and change event scheduling
- Provides monitoring and metrics
- Coordinates with core components
- Maintains execution guarantees

Design Patterns:
- State Pattern for execution
- Publisher/Subscriber for monitoring
- Factory Pattern for creation
- Builder Pattern for configuration
- Singleton Pattern for schedulers

Security:
- Execution isolation
- Resource monitoring
- Timer management
- Event validation
- Extension boundaries

Cross-cutting:
- Error handling with recovery
- Performance optimization
- Execution metrics
- Thread safety
"""

from .executor import Executor
from .scheduler import Scheduler
from .monitor import Monitor

__all__ = ["Executor", "Scheduler", "Monitor"]
