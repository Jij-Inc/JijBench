from __future__ import annotations
from dataclasses import dataclass, field

import copy
import dill
import numpy as np
import pandas as pd
import typing as tp
import jijmodeling as jm
import pathlib
import uuid


DNodeInType = tp.TypeVar("DNodeInType", bound="DataNode")
DNodeOutType = tp.TypeVar("DNodeOutType", bound="DataNode")

DEFAULT_RESULT_DIR = pathlib.Path("./.jb_results")


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


class RecordFactory(FunctionNode["DataNode", "Record"]):
    def __call__(self, inputs: list[DataNode], name: str | None = None) -> Record:
        data = pd.Series({node.name: node.data for node in inputs})
        return Record(data, name=name)

    @property
    def name(self) -> str:
        return "record"


class TableFactory(FunctionNode["Record", "Table"]):
    def __call__(
        self,
        inputs: list[Record],
        name: str | None = None,
        index_name: str | None = None,
    ) -> Table:
        data = pd.DataFrame({node.name: node.data for node in inputs}).T
        data.index.name = index_name
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
class DataBase(DataNode):
    def append(self, record: Record, **kwargs: tp.Any) -> None:
        raise NotImplementedError

    def _append(
        self, record: Record, factory: TableFactory | ArtifactFactory, **kwargs: tp.Any
    ) -> None:
        node = factory.apply([record], name=self.name)
        node.operator = factory

        c = Concat()
        inputs = [copy.deepcopy(self), node]
        c.inputs = inputs
        self.data = c(inputs, **kwargs).data
        self.operator = c


@dataclass
class Table(DataBase):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def append(
        self,
        record: Record,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
        **kwargs: tp.Any,
    ) -> None:
        self._append(record, TableFactory(), axis=axis, index_name=index_name)


@dataclass
class Artifact(DataBase):
    data: dict = field(default_factory=dict)

    def append(self, record: Record, **kwargs: tp.Any) -> None:
        self._append(record, ArtifactFactory(), **kwargs)


class Concat(FunctionNode["DataBase", "DataBase"]):
    def __call__(
        self,
        inputs: list[DataBase],
        name: str | None = None,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> DataBase:
        dtype = type(inputs[0])
        if not all([isinstance(node, dtype) for node in inputs]):
            raise TypeError(
                "Type of elements of 'inputs' must be unified with either 'Table' or 'Artifact'."
            )

        if isinstance(inputs[0], Artifact):
            data = inputs[0].data.copy()
            for node in inputs[1:]:
                if node.name in data:
                    data[node.name].update(node.data.copy())
                else:
                    data[node.name] = node.data.copy()
            return Artifact(data=data, name=name)
        elif isinstance(inputs[0], Table):
            data = pd.concat([node.data for node in inputs], axis=axis)
            data.index.name = index_name
            return Table(data=data, name=name)
        else:
            raise TypeError(f"'{inputs[0].__class__.__name__}' type is not supported.")

    @property
    def name(self) -> str:
        return "concat"


@dataclass
class Experiment(DataBase):
    def __init__(
        self,
        data: tuple[Artifact, Table] | None = None,
        name: str | None = None,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ):
        if name is None:
            name = ID().data

        if data is None:
            data = (Artifact(), Table())

        if data[0].name is None:
            data[0].name = name

        if data[1].name is None:
            data[1].name = name

        self.data = data
        self.name = name
        self.autosave = autosave

        if isinstance(savedir, str):
            savedir = pathlib.Path(savedir)
        self.savedir = savedir

    @property
    def artifact(self) -> dict:
        return self.data[0].data

    @property
    def table(self) -> pd.DataFrame:
        t = self.data[1].data
        is_tuple_index = all([isinstance(i, tuple) for i in t.index])
        if is_tuple_index:
            names = t.index.names if len(t.index.names) >= 2 else None
            index = pd.MultiIndex.from_tuples(t.index, names=names)
            t.index = index
        return t

    def __enter__(self) -> Experiment:
        p = self.savedir / str(self.name)
        (p / "table").mkdir(parents=True, exist_ok=True)
        (p / "artifact").mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        index = (self.name, self.table.index[-1])
        self.table.rename(index={self.table.index[-1]: index}, inplace=True)

        if self.autosave:
            self.save()

    def append(self, record: Record) -> None:
        for d in self.data:
            d.append(record, index_name=("experiment_id", "run_id"))

    def save(self):
        def is_dillable(obj: tp.Any):
            try:
                dill.dumps(obj)
                return True
            except Exception:
                return False

        p = self.savedir / str(self.name) / "table" / "table.csv"
        self.table.to_csv(p)

        p = self.savedir / str(self.name) / "artifact" / "artifact.dill"
        record_name = list(self.data[0].operator.inputs[1].data.keys())[0]
        if p.exists():
            with open(p, "rb") as f:
                artifact = dill.load(f)
                artifact[self.name][record_name] = {}
        else:
            artifact = {self.name: {record_name: {}}}

        record = {}
        for k, v in self.artifact[self.name][record_name].items():
            if is_dillable(v):
                record[k] = v
            else:
                record[k] = str(v)
        artifact[self.name][record_name].update(record)

        with open(p, "wb") as f:
            dill.dump(artifact, f)


class Solver(FunctionNode):
    def __init__(self, f: tp.Callable) -> None:
        super().__init__()
