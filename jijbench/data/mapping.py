from __future__ import annotations

import pandas as pd
import typing as tp

from dataclasses import dataclass, field
from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, TableFactory
from jijbench.node.base import DataNode
from jijbench.typing import MappingTypes
from typing_extensions import TypeGuard

if tp.TYPE_CHECKING:
    from jijbench.experiment.experiment import Experiment


def _is_artifact(
    node: MappingTypes,
) -> TypeGuard[Artifact]:
    return node.__class__.__name__ == "Artifact"


def _is_experiment(
    node: MappingTypes,
) -> TypeGuard[Experiment]:
    return node.__class__.__name__ == "Experiment"


def _is_record(
    node: MappingTypes,
) -> TypeGuard[Record]:
    return node.__class__.__name__ == "Record"


def _is_table(
    node: MappingTypes,
) -> TypeGuard[Table]:
    return node.__class__.__name__ == "Table"


@dataclass
class Mapping(DataNode):
    @tp.overload
    def append(self, record: Record) -> None:
        ...

    @tp.overload
    def append(
        self,
        record: Record,
        axis: tp.Literal[0, 1] = 0,
        index_name: tp.Any | None = None,
    ):
        ...

    @tp.overload
    def append(self, record: Record) -> None:
        ...

    @tp.overload
    def append(
        self,
        record: Record,
        axis: tp.Literal[0, 1] = 0,
        index_name: tp.Any | None = None,
    ) -> None:
        ...

    def append(
        self,
        record: Record,
        axis: tp.Literal[0, 1] = 0,
        index_name: tp.Any | None = None,
    ) -> None:
        concat = Concat()
        node = tp.cast(MappingTypes, self)
        if _is_artifact(node):
            others = [ArtifactFactory().operate([record])]
            node.data = node.apply(concat, others).data
        elif _is_experiment(node):
            node.data[0].append(record)
            node.data[1].append(record, axis, index_name)
        elif _is_record(node):
            node.data = node.apply(concat, [record]).data
        elif _is_table(node):
            others = [TableFactory().operate([record])]
            node.data = node.apply(concat, others).data
        else:
            raise TypeError(f"{node.__class__.__name__} does not support 'append'.")
        node.operator = concat


@dataclass
class Record(Mapping):
    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))
    name: str | None = None


@dataclass
class Artifact(Mapping):
    data: dict = field(default_factory=dict)
    name: str | None = None


@dataclass
class Table(Mapping):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    name: str | None = None
