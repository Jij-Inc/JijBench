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
RecordOr
FNodeType = tp.TypeVar("FNodeType", bound="FunctionNode")


@dataclass
class DataNode:
    data: tp.Any
    name: str | None = None
    operator: FunctionNode | None = None


class FunctionNode(tp.Generic[DNodeInType, DNodeOutType]):
    def __init__(
        self,
    ) -> None:
        self.inputs: list[DataNode] = []

    @property
    def name(self) -> str | None:
        raise NotImplementedError

    def apply(self, inputs: list[DNodeInType], **kwargs: tp.Any) -> DNodeOutType:
        self.inputs += inputs
        return self.operate(inputs, **kwargs)

    def operate(self, inputs: list[DNodeInType], **kwargs: tp.Any) -> DNodeOutType:
        raise NotImplementedError


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

    def min(self) -> Value:
        return Min().apply([self])

    def max(self) -> DataNode:
        return Max().apply([self])

    def mean(self) -> DataNode:
        return Mean().apply([self])

    def std(self) -> DataNode:
        return Std().apply([self])


class Min(FunctionNode["Array", "Value"]):
    @property
    def name(self) -> str:
        return "min"

    def apply(self, inputs: list[Array], **kwargs: tp.Any) -> Value:
        return self.operate(inputs, **kwargs)

    def operate(self, inputs: list[Array]) -> Value:
        data = inputs[0].data.min()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = Value(data=data, name=name)
        node.operator = self
        return node


class Max(FunctionNode["Array", "Value"]):
    @property
    def name(self) -> str:
        return "max"

    def operate(self, inputs: list[Array]) -> Value:
        data = inputs[0].data.max()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = Value(data=data, name=name, operator=self)
        return node


class Mean(FunctionNode["Array", "Value"]):
    @property
    def name(self) -> str:
        return "mean"

    def operate(self, inputs: list[Array]) -> Value:
        data = inputs[0].data.mean()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = Value(data=data, name=name, operator=self)
        return node


class Std(FunctionNode["Array", "Value"]):
    @property
    def name(self) -> str:
        return "std"

    def operate(self, inputs: list[Array]) -> Value:
        data = inputs[0].data.std()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = Value(data=data, name=name, operator=self)
        return node


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
    @property
    def name(self) -> str:
        return "record"

    def operate(self, inputs: list[DNodeInType], name: str | None = None) -> Record:
        data = pd.Series({node.name: node.data for node in inputs})
        return Record(data, name=name, operator=self)


class TableFactory(FunctionNode["Record", "Table"]):
    @property
    def name(self) -> str:
        return "table"

    def operate(self, inputs: list[Record], name: str | None = None) -> Table:
        data = pd.DataFrame({node.name: node.data for node in inputs})
        return Table(data, name=name, operator=self)


class ArtifactFactory(FunctionNode["Record", "Artifact"]):
    @property
    def name(self) -> str:
        return "artifact"

    def operate(self, inputs: list[Record], name: str | None = None) -> Artifact:
        data = {node.name: node.data.to_dict() for node in inputs}
        return Artifact(data, name=name, operator=self)


@dataclass
class Record(DataNode):
    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))


@dataclass
class Result(DataNode):
    pass


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


class Concat(FunctionNode[DNodeInType, DNodeOutType]):
    name = "concat"

    def operate(
        self, inputs: list[DNodeInType], name: str | None = None, axis: tp.Literal[0, 1] = 0
    ) -> DataNode[Concat]:
        dtype = type(inputs[0])
        if not all([isinstance(node, dtype) for node in inputs]):
            raise TypeError(
                "Type of elements of 'inputs' must be unified with either 'Table' or 'Artifact'."
            )

        if isinstance(inputs[0], Artifact):
            # data = {node.name: node.data for node in inputs}
            data = {}
            for n in [1, 2, 3]:
                pass
            # for node in inputs:
            #     if node.name in data:
            #         data[node.name].update(node.data)
            #     else:
            #         data[node.name] = node.data
            return Artifact(data=data, name=name, operator=self)
        elif isinstance(inputs[0], Table):
            data = pd.concat(
                [node.data for node in inputs if isinstance(node, Table)], axis=axis
            )
            return Table(data=data, name=name, operator=self)
        else:
            raise TypeError(f"'{inputs[0].__class__.__name__}' type is not supported.")


@dataclass
class Experiment(DataNode):
    data: tuple = ()

    def __post_init__(self):
        if not self.data:
            self.data = (Table(), Artifact())
