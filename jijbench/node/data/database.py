from __future__ import annotations

import copy
import typing as tp

from dataclasses import dataclass
from jijbench.node.base import DataNode
import jijbench.node.functions.concat as _concat

if tp.TYPE_CHECKING:
    from jijbench.node.data.record import Record
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

        c = _concat.Concat()
        inputs = [copy.deepcopy(self), node]
        c.inputs = inputs
        self.data = c(inputs, **kwargs).data
        self.operator = c
