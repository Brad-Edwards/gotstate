"""
Persistence package for storage and validation.

Architecture:
- Manages state machine persistence
- Handles definition validation
- Maintains version compatibility
- Coordinates with core components
- Ensures data integrity

Design Patterns:
- Strategy Pattern for storage
- Builder Pattern for loading
- Visitor Pattern for validation
- Chain of Responsibility for rules
- Observer Pattern for changes

Security:
- Data validation
- Format verification
- Version checking
- Resource protection
- Access control

Cross-cutting:
- Error handling
- Performance optimization
- Storage metrics
- Thread safety
"""

from .serializer import Serializer
from .validator import Validator

__all__ = ["Serializer", "Validator"]
