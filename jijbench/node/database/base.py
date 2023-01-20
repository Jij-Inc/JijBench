from __future__ import annotations

import copy
import typing as tp

from dataclasses import dataclass
from jijbench.node import DataNode

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
