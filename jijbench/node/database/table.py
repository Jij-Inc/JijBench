from __future__ import annotations

import pandas as pd
import typing as tp

from dataclasses import dataclass, field
from jijbench.node import FunctionNode


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
