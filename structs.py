"""Useful classes."""
from dataclasses import dataclass
from typing import List


@dataclass
class ObjectType:
    """Represents a type of object (e.g., 'cup', 'table', 'person')."""
    name: str


@dataclass
class Object:
    """Represents a specific object with a name and type."""
    unique_id: str
    name: str
    type: ObjectType


@dataclass
class Predicate:
    """Represents a predicate with a name and list of object arguments."""
    name: str
    object_args: List[Object]





