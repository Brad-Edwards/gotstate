"""
Core type system definitions and management.

Defines the core type system for the state machine, including
base types, type constraints, and type operations.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from enum import Enum, auto
from typing import Any, Dict, Generic, List, Optional, Set, TypeVar

import icontract

T = TypeVar("T")


class TypeKind(Enum):
    """Defines the different kinds of types."""

    PRIMITIVE = auto()
    COMPOSITE = auto()
    GENERIC = auto()
    UNION = auto()
    EXTENSION = auto()


class TypeConstraint(Enum):
    """Defines type system constraints."""

    IMMUTABLE = auto()
    COVARIANT = auto()
    INVARIANT = auto()
    BOUNDED = auto()


class BaseType(ABC):
    """Base class for all types in the system.

    Defines common type behavior and operations via the
    Template Method pattern.
    """

    @icontract.require(lambda name: isinstance(name, str) and len(name) > 0, "Name must be a non-empty string")
    @icontract.require(lambda kind: isinstance(kind, TypeKind), "Kind must be a valid TypeKind")
    def __init__(self, name: str, kind: TypeKind) -> None:
        self._name = name
        self._kind = kind
        self._constraints: Set[TypeConstraint] = set()
        self._metadata: Dict[str, Any] = {}

    @property
    def name(self) -> str:
        return self._name

    @property
    def kind(self) -> TypeKind:
        return self._kind

    @property
    def constraints(self) -> Set[TypeConstraint]:
        return frozenset(self._constraints)

    def add_constraint(self, constraint: TypeConstraint) -> None:
        self._constraints.add(constraint)

    @abstractmethod
    def is_compatible(self, other: BaseType) -> bool:
        """Check if this type is compatible with another type."""
        ...

    @abstractmethod
    def validate(self, value: Any) -> bool:
        """Validate that a value conforms to this type."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self._name!r}, kind={self._kind.name})"


class PrimitiveType(BaseType):
    """Represents primitive types (int, str, float, bool, etc.)."""

    def __init__(self, name: str, python_type: type) -> None:
        super().__init__(name, TypeKind.PRIMITIVE)
        self._python_type = python_type
        self.add_constraint(TypeConstraint.IMMUTABLE)

    @property
    def python_type(self) -> type:
        return self._python_type

    def is_compatible(self, other: BaseType) -> bool:
        if isinstance(other, PrimitiveType):
            return issubclass(self._python_type, other._python_type) or issubclass(
                other._python_type, self._python_type
            )
        return False

    def validate(self, value: Any) -> bool:
        return isinstance(value, self._python_type)


class CompositeType(BaseType):
    """Represents composite types composed of other types."""

    def __init__(self, name: str) -> None:
        super().__init__(name, TypeKind.COMPOSITE)
        self._fields: Dict[str, BaseType] = {}

    @property
    def fields(self) -> Dict[str, BaseType]:
        return dict(self._fields)

    def add_field(self, name: str, field_type: BaseType) -> None:
        self._fields[name] = field_type

    def is_compatible(self, other: BaseType) -> bool:
        if not isinstance(other, CompositeType):
            return False
        for name, ftype in self._fields.items():
            if name not in other._fields:
                return False
            if not ftype.is_compatible(other._fields[name]):
                return False
        return True

    def validate(self, value: Any) -> bool:
        if not isinstance(value, dict):
            return False
        for name, ftype in self._fields.items():
            if name not in value:
                return False
            if not ftype.validate(value[name]):
                return False
        return True


class GenericType(BaseType, Generic[T]):
    """Represents generic (parameterized) types."""

    def __init__(self, name: str, bound: Optional[BaseType] = None) -> None:
        super().__init__(name, TypeKind.GENERIC)
        self._bound = bound
        if bound is not None:
            self.add_constraint(TypeConstraint.BOUNDED)

    @property
    def bound(self) -> Optional[BaseType]:
        return self._bound

    def is_compatible(self, other: BaseType) -> bool:
        if self._bound is not None:
            return self._bound.is_compatible(other)
        return True

    def validate(self, value: Any) -> bool:
        if self._bound is not None:
            return self._bound.validate(value)
        return True


class UnionType(BaseType):
    """Represents union types (one of several possible types)."""

    def __init__(self, name: str, variants: Optional[List[BaseType]] = None) -> None:
        super().__init__(name, TypeKind.UNION)
        self._variants: List[BaseType] = list(variants) if variants else []

    @property
    def variants(self) -> List[BaseType]:
        return list(self._variants)

    def add_variant(self, variant: BaseType) -> None:
        self._variants.append(variant)

    def is_compatible(self, other: BaseType) -> bool:
        return any(v.is_compatible(other) for v in self._variants)

    def validate(self, value: Any) -> bool:
        return any(v.validate(value) for v in self._variants)


class TypeRegistry:
    """Manages type registration and lookup."""

    def __init__(self) -> None:
        self._types: Dict[str, BaseType] = {}
        self._lock = threading.Lock()

    def register(self, type_def: BaseType) -> None:
        with self._lock:
            self._types[type_def.name] = type_def

    def get(self, name: str) -> Optional[BaseType]:
        with self._lock:
            return self._types.get(name)

    def all_types(self) -> Dict[str, BaseType]:
        with self._lock:
            return dict(self._types)
