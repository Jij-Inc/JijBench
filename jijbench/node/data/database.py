from __future__ import annotations

import copy
import pandas as pd
import typing as tp

from dataclasses import dataclass, field
from jijbench.node.base import DataNode
from jijbench.node.data.record import Record
import jijbench.node.functions.concat

if tp.TYPE_CHECKING:
    from jijbench.node.functions.factory import ArtifactFactory, TableFactory


@dataclass
class DataBase(DataNode):
    def append(self, record: Record, **kwargs: tp.Any) -> None:
        raise NotImplementedError

    def _append(
        self, record: Record, factory: TableFactory | ArtifactFactory, **kwargs: tp.Any
    ) -> None:
        node = factory.apply([record], name=self.name)
        node.operator = factory

        from jijbench.node.functions.concat import Concat

        c = Concat()
        inputs = [copy.deepcopy(self), node]
        c.inputs = inputs
        self.data = c(inputs, **kwargs).data
        self.operator = c


@dataclass
class Artifact(DataBase):
    data: dict = field(default_factory=dict)

    def append(self, record: Record, **kwargs: tp.Any) -> None:
        from jijbench.node.functions.factory import ArtifactFactory

        self._append(record, ArtifactFactory(), **kwargs)


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
        from jijbench.node.functions.factory import TableFactory

        self._append(record, TableFactory(), axis=axis, index_name=index_name)
