"""
Types package for type system integration.

Architecture:
- Defines core type system
- Manages type extensions
- Maintains type safety
- Coordinates with validation
- Ensures type consistency

Design Patterns:
- Factory Pattern: Type creation
- Strategy Pattern: Type handling
- Adapter Pattern: Type conversion
- Visitor Pattern: Type validation
- Composite Pattern: Type composition

Security:
- Type validation
- Conversion safety
- Extension isolation
- Resource protection
- Access control

Cross-cutting:
- Error handling
- Performance optimization
- Type metrics
- Thread safety
"""

from .base import BaseType
from .extensions import TypeExtension

__all__ = ["BaseType", "TypeExtension"]
