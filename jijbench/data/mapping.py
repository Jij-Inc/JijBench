from __future__ import annotations

import abc
import pandas as pd

from dataclasses import dataclass, field
from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, TableFactory
from jijbench.node.base import DataNode
from jijbench.typing import T


@dataclass
class Mapping(DataNode[T]):
    @abc.abstractmethod
    def append(self, record: Record) -> None:
        pass

    @abc.abstractmethod
    def view(self) -> T:
        pass


@dataclass
class Record(Mapping):
    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))
    name: str | None = None

    def append(self, record: Record) -> None:
        concat: Concat[Record] = Concat()
        node = self.apply(concat, [record], name=self.name)
        self.__init__(**node.__dict__)

    def view(self) -> pd.Series:
        return self.data.apply(lambda x: x.data)


@dataclass
class Artifact(Mapping[T]):
    data: dict[str, dict[str, DataNode[T]]] = field(default_factory=dict)
    name: str | None = None

    def append(self, record: Record) -> None:
        concat: Concat[Artifact] = Concat()
        other = ArtifactFactory()([record])
        node = self.apply(concat, [other], name=self.name)
        self.__init__(**node.__dict__)

    def view(self) -> dict[str, dict[str, T]]:
        return {
            k: {name: node.data for name, node in v.items()}
            for k, v in self.data.items()
        }


@dataclass
class Table(Mapping):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    name: str | None = None

    def append(self, record: Record) -> None:
        concat: Concat[Table] = Concat()
        other = TableFactory()([record])
        node = self.apply(concat, [other], name=self.name, axis=0)
        self.__init__(**node.__dict__)

    def view(self) -> pd.DataFrame:
        if self.data.empty:
            return self.data
        else:
            is_tuple_index = all([isinstance(i, tuple) for i in self.data.index])
            if is_tuple_index:
                names = self.data.index.names if len(self.data.index.names) >= 2 else None
                index = pd.MultiIndex.from_tuples(self.data.index, names=names)
                # TODO 代入しない
                self.data.index = index
            return self.data.applymap(lambda x: x.data)
