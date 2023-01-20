from __future__ import annotations

import copy
import pandas as pd
import typing as tp

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from jijbench.node.base import DataNode
from jijbench.data.record import Record

if tp.TYPE_CHECKING:
    from jijbench.functions.factory import ArtifactFactory, TableFactory


@dataclass
class Mapping(DataNode, metaclass=ABCMeta):
    @abstractmethod
    def append(self, record: Record, **kwargs: tp.Any) -> None:
        pass

    def _append(
        self,
        record: Record,
        factory: TableFactory | ArtifactFactory,
        **kwargs: tp.Any,
    ) -> None:
        from jijbench.functions.concat import Concat

        node = record.apply(factory, name=self.name)

        concat = Concat()
        inputs = [copy.deepcopy(self), node]
        self.data = concat(inputs, **kwargs).data
        self.operator = c


@dataclass
class Artifact(Mapping):
    data: dict = field(default_factory=dict)

    def append(self, record: Record, **kwargs: tp.Any) -> None:
        from jijbench.functions.factory import ArtifactFactory

        self._append(record, ArtifactFactory(), **kwargs)


@dataclass
class Table(Mapping):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def append(
        self,
        record: Record,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
        **kwargs: tp.Any,
    ) -> None:
        from jijbench.functions.factory import TableFactory

        self._append(record, TableFactory(), axis=axis, index_name=index_name)
