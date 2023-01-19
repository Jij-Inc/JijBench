from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import typing as tp
import jijmodeling as jm
import pathlib
import uuid

DNodeInType = tp.TypeVar("DNodeInType", bound="DataNode")
DNodeOutType = tp.TypeVar("DNodeOutType", bound="DataNode")
ConcatInType = tp.TypeVar("ConcatInType", "Artifact", "Table")
ConcatOutType = tp.TypeVar("ConcatOutType", "Artifact", "Table")
# FNodeType = tp.TypeVar("FNodeType", bound="FunctionNode")


@dataclass
class DataNode:
    data: tp.Any
    name: str | None = None

    def __post_init__(self) -> None:
        self.operator: FunctionNode | None = None


class FunctionNode(tp.Generic[DNodeInType, DNodeOutType]):
    def __init__(
        self,
    ) -> None:
        self.inputs: list[DataNode] = []

    def __call__(self, inputs: list[DNodeInType], **kwargs: tp.Any) -> DNodeOutType:
        raise NotImplementedError

    @property
    def name(self) -> str | None:
        raise NotImplementedError

    def apply(self, inputs: list[DNodeInType], **kwargs: tp.Any) -> DNodeOutType:
        self.inputs += inputs
        node = self(inputs, **kwargs)
        node.operator = self
        return node


DEFAULT_RESULT_DIR = "./.jb_results"


@dataclass
class ID(DataNode):
    data: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __post_init__(self) -> None:
        self.data = str(self.data)


@dataclass
class Date(DataNode):
    data: str | pd.Timestamp = field(default_factory=pd.Timestamp.now)
    name: str = "timestamp"

    def __post_init__(self) -> None:
        if isinstance(self.data, str):
            self.data = pd.Timestamp(self.data)


@dataclass
class Value(DataNode):
    data: int | float


@dataclass
class Array(DataNode):
    data: np.ndarray

    def min(self) -> Array:
        return Min().apply([self])

    def max(self) -> Array:
        return Max().apply([self])

    def mean(self) -> Array:
        return Mean().apply([self])

    def std(self) -> Array:
        return Std().apply([self])


class Min(FunctionNode["Array", "Array"]):
    def __call__(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.min()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "min"


class Max(FunctionNode["Array", "Array"]):
    def __call__(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.max()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "max"


class Mean(FunctionNode["Array", "Array"]):
    def __call__(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.mean()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "mean"


class Std(FunctionNode["Array", "Array"]):
    def __call__(self, inputs: list[Array]) -> Array:
        data = inputs[0].data.std()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = Array(data=data, name=name)
        return node

    @property
    def name(self) -> str:
        return "std"


@dataclass
class Energy(Array):
    name: str = "energy"


@dataclass
class Objective(Array):
    name: str = "objective"


@dataclass
class ConstraintViolation(Array):
    def __post_init__(self):
        if self.name is None:
            raise NameError("Attribute 'name' is None. Please set a name.")


@dataclass
class SampleSet(DataNode):
    data: jm.SampleSet


class RecordFactory(FunctionNode[DNodeInType, "Record"]):
    def __call__(self, inputs: list[DNodeInType], name: str | None = None) -> Record:
        data = pd.Series({node.name: node.data for node in inputs})
        return Record(data, name=name)

    @property
    def name(self) -> str:
        return "record"


class TableFactory(FunctionNode["Record", "Table"]):
    def __call__(self, inputs: list[Record], name: str | None = None) -> Table:
        data = pd.DataFrame({node.name: node.data for node in inputs})
        return Table(data, name=name)

    @property
    def name(self) -> str:
        return "table"


class ArtifactFactory(FunctionNode["Record", "Artifact"]):
    def __call__(self, inputs: list[Record], name: str | None = None) -> Artifact:
        data = {node.name: node.data.to_dict() for node in inputs}
        return Artifact(data, name=name)

    @property
    def name(self) -> str:
        return "artifact"


@dataclass
class Record(DataNode):
    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))


@dataclass
class Table(DataNode):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def append(self, record: Record, axis: tp.Literal[0, 1] = 0) -> Table:
        table = TableFactory().apply([record], name=self.name)
        return Concat().apply([self, table], axis=axis)


@dataclass
class Artifact(DataNode):
    data: dict = field(default_factory=dict)

    def append(self, record: Record) -> Artifact:
        artifact = ArtifactFactory().apply([record], name=self.name)
        return Concat().apply([self, artifact])


class Concat(FunctionNode[ConcatInType, ConcatOutType]):
    def __call__(
        self,
        inputs: list[Artifact] | list[Table],
        name: str | None = None,
        axis: tp.Literal[0, 1] = 0,
    ) -> Artifact | Table:
        dtype = type(inputs[0])
        if not all([isinstance(node, dtype) for node in inputs]):
            raise TypeError(
                "Type of elements of 'inputs' must be unified with either 'Table' or 'Artifact'."
            )

        if isinstance(inputs[0], Artifact):
            data = {}
            for node in inputs:
                if node.name in data:
                    data[node.name].update(node.data)
                else:
                    data[node.name] = node.data
            return Artifact(data=data, name=name)
        elif isinstance(inputs[0], Table):
            data = pd.concat(
                [node.data for node in inputs if isinstance(node, Table)], axis=axis
            )
            return Table(data=data, name=name)
        else:
            raise TypeError(f"'{inputs[0].__class__.__name__}' type is not supported.")

    @property
    def name(self) -> str:
        return "concat"


@dataclass
class Experiment(DataNode):
    data: tuple = ()

    def __post_init__(self):
        if not self.data:
            self.data = (Table(), Artifact())
