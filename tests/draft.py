from __future__ import annotations
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import typing as tp
import jijmodeling as jm
import pathlib
import uuid

T_in = tp.TypeVar("T_in", bound="DataNode", covariant=True)
T_out = tp.TypeVar("T_out", bound="DataNode", covariant=True)
T_f = tp.TypeVar("T_f", bound="FunctionNode", covariant=True)
T_c = tp.TypeVar("T_c", "Artifact", "Table")


@dataclass
class DataNode(tp.Generic[T_f]):
    data: tp.Any
    name: str | None = None
    operator: T_f | None = None


class FunctionNode(tp.Generic[T_in, T_out]):
    name: str | None = None

    def __init__(
        self,
    ) -> None:
        self.inputs = []

    def apply(self, inputs: list[T_in], **kwargs: tp.Any) -> T_out:
        self.inputs += inputs
        return self.operate(inputs, **kwargs)

    def operate(self, inputs: list[T_in], **kwargs: tp.Any) -> T_out:
        raise NotImplementedError


DEFAULT_RESULT_DIR = "./.jb_results"


@dataclass
class ID(DataNode):
    data: str | uuid.UUID = field(default_factory=uuid.uuid4)

    def __post_init__(self):
        self.data = str(self.data)


@dataclass
class Date(DataNode):
    data: str | pd.Timestamp = field(default_factory=pd.Timestamp.now)
    name: str = "timestamp"

    def __post_init__(self):
        if isinstance(self.data, str):
            self.data = pd.Timestamp(self.data)


class Min(FunctionNode["Array", "DataNode[Min]"]):
    name = "min"

    def operate(self, inputs: list[Array]) -> DataNode[Min]:
        data = inputs[0].data.min()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = DataNode(data=data, name=name, operator=self)
        return node


class Max(FunctionNode["Array", "DataNode[Max]"]):
    name = "max"

    def operate(self, inputs: list[Array]) -> DataNode[Max]:
        data = inputs[0].data.max()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = DataNode(data=data, name=name, operator=self)
        return node


class Mean(FunctionNode["Array", "DataNode[Mean]"]):
    name = "mean"

    def operate(self, inputs: list[Array]) -> DataNode:
        data = inputs[0].data.mean()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = DataNode(data=data, name=name, operator=self)
        return node


class Std(FunctionNode["Array", "DataNode[Std]"]):
    name = "std"

    def operate(self, inputs: list[Array]) -> DataNode[Std]:
        data = inputs[0].data.std()
        name = inputs[0].name + f"_{self.name}" if inputs[0].name else None
        node = DataNode(data=data, name=name, operator=self)
        return node


@dataclass
class Array(DataNode):
    data: np.ndarray

    def min(self) -> DataNode:
        return Min().apply([self])

    def max(self) -> DataNode:
        return Max().apply([self])

    def mean(self) -> DataNode:
        return Mean().apply([self])

    def std(self) -> DataNode:
        return Std().apply([self])


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


class RecordFactory(FunctionNode["T_in", "Record"]):
    name = "record"

    def operate(self, inputs: list[T_in], name: str | None = None) -> Record:
        data = pd.Series({node.name: node.data for node in inputs})
        return Record(data, name=name, operator=self)


class TableFactory(FunctionNode["Record", "Table"]):
    name = "table"

    def operate(self, inputs: list[Record], name: str | None = None) -> Table:
        data = pd.DataFrame({node.name: node.data for node in inputs})
        return Table(data, name=name, operator=self)


class ArtifactFactory(FunctionNode["Record", "Artifact"]):
    name = "artifact"

    def operate(self, inputs: list[Record], name: str | None = None) -> Artifact:
        data = {node.name: node.data.to_dict() for node in inputs}
        return Artifact(data, name=name, operator=self)


@dataclass
class Record(DataNode):
    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))


@dataclass
class Table(DataNode):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def append(self, record: Record, axis: tp.Literal[0, 1] = 0) -> Table:
        table = TableFactory().apply([record])
        return Concat().apply([self, table], axis=axis)


@dataclass
class Artifact(DataNode):
    data: dict = field(default_factory=dict)


class Concat(FunctionNode[T_c, T_c]):
    name = "concat"

    def operate(self, inputs: list[T_c], name=None, axis: tp.Literal[0, 1] = 0) -> T_c:
        dtype = type(inputs[0])
        if not all([isinstance(node, dtype) for node in inputs]):
            raise TypeError(
                "Type of elements of 'inputs' must be unified with either 'Table' or 'Artifact'."
            )

        if isinstance(inputs[0], Artifact):
            data = {node.name: node.data for node in inputs}
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
