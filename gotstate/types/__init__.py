"""
Types package for type system integration.

Defines the core type system, manages type extensions,
and maintains type safety.
"""

from .base import (
    BaseType,
    CompositeType,
    GenericType,
    PrimitiveType,
    TypeConstraint,
    TypeKind,
    TypeRegistry,
    UnionType,
)
from .extensions import ExtensionManager, ExtensionScope, ExtensionStatus, TypeExtension

__all__ = [
    "BaseType",
    "TypeKind",
    "TypeConstraint",
    "PrimitiveType",
    "CompositeType",
    "GenericType",
    "UnionType",
    "TypeRegistry",
    "TypeExtension",
    "ExtensionStatus",
    "ExtensionScope",
    "ExtensionManager",
]
