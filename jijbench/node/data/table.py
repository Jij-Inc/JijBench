from __future__ import annotations

import pandas as pd
import typing as tp

from dataclasses import dataclass, field
import jijbench.node.data.database as _dataBase
import jijbench.node.functions.factory as _factory

if tp.TYPE_CHECKING:
    from jijbench.node.data.record import Record


@dataclass
class Table(_dataBase.DataBase):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def append(
        self,
        record: Record,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
        **kwargs: tp.Any,
    ) -> None:
        self._append(record, _factory.TableFactory(), axis=axis, index_name=index_name)
