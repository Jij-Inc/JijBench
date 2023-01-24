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
    def append(self: Mapping, record: Record, **kwargs: tp.Any) -> None:
        pass

    def _append(
        self,
        record: Record,
        factory: TableFactory | ArtifactFactory,
        **kwargs: tp.Any,
    ) -> None:
        append(self, record, **kwargs)


@dataclass
class Artifact(Mapping):
    data: dict = field(default_factory=dict)



@dataclass
class Table(Mapping):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)

    def append(
        self,
        record: Record,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> None:
        from jijbench.functions.factory import TableFactory

        self._append(record, TableFactory(), axis=axis, index_name=index_name)


@tp.overload
def append(
    mapping: Artifact,
    record: Record,
) -> None:
    ...


@tp.overload
def append(
    mapping: Table,
    record: Record,
    axis: tp.Literal[0, 1] = 0,
    index_name: str | None = None,
) -> None:
    ...


def append(
    mapping: Mapping,
    record: Record,
    axis: tp.Literal[0, 1] = 0,
    index_name: str | None = None,
) -> None:
    from jijbench.functions.concat import Concat
    from jijbench.functions.factory import TableFactory, ArtifactFactory

    if isinstance(mapping, Artifact):
        node = record.apply(ArtifactFactory(), name=mapping.name)
    elif isinstance(mapping, Table):
        node = record.apply(TableFactory(), name=mapping.name)
    else:
        raise TypeError(f"{mapping.__class__.__name__} does not support 'append'.")
    concat = Concat()
    mapping.data = mapping.apply(concat, [node], axis=axis, index_name=index_name).data
    mapping.operator = concat
