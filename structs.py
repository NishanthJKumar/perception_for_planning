from dataclasses import dataclass


@dataclass
class ObjectType:
    name: str


@dataclass
class Object:
    unique_id: str
    name: str
    type: ObjectType


@dataclass
class Predicate:
    name: str
    arg_types: list[ObjectType]
    description: str


@dataclass
class Atom:
    name: str
    object_args: list[Object]





