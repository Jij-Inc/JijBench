from __future__ import annotations

import abc
import pandas as pd

from dataclasses import dataclass, field
from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, TableFactory
from jijbench.node.base import DataNode


@dataclass
class Mapping(DataNode):
    @abc.abstractmethod
    def append(self, record: Record) -> None:
        pass


@dataclass
class Record(Mapping):
    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))
    name: str | None = None

    def append(self, record: Record) -> None:
        concat: Concat[Record] = Concat()
        node = self.apply(concat, [record])
        self.data = node.data
        self.operator = node.operator


@dataclass
class Artifact(Mapping):
    data: dict = field(default_factory=dict)
    name: str | None = None

    def append(self, record: Record) -> None:
        concat: Concat[Artifact] = Concat()
        other = ArtifactFactory()([record])
        node = self.apply(concat, [other])
        self.data = node.data
        self.operator = node.operator


@dataclass
class Table(Mapping):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    name: str | None = None

    def append(self, record: Record) -> None:
        concat: Concat[Table] = Concat()
        other = TableFactory()([record])
        node = self.apply(concat, [other], axis=0)
        self.data = node.data
        self.operator = node.operator
