from __future__ import annotations

import abc
import pandas as pd
import typing as tp

from abc import ABCMeta, abstractmethod
from dataclasses import dataclass, field
from jijbench.node.base import DataNode
from typing_extensions import TypeGuard

if tp.TYPE_CHECKING:
    from jijbench.experiment.experiment import Experiment


def _is_artifact(
    node: Mapping,
) -> TypeGuard[Artifact]:
    return node.__class__.__name__ == "Artifact"


def _is_experiment(
    node: Mapping,
) -> TypeGuard[Experiment]:
    return node.__class__.__name__ == "Experiment"


def _is_record(
    node: Mapping,
) -> TypeGuard[Record]:
    return node.__class__.__name__ == "Record"


def _is_table(
    node: Mapping,
) -> TypeGuard[Table]:
    return node.__class__.__name__ == "Table"


@dataclass
class Mapping(DataNode):
    @abc.abstractmethod
    def append(self, record: Record) -> None:
        pass

    # @tp.overload
    # def append(self, record: Record) -> None:
    #     ...


#
# @tp.overload
# def append(
#     self,
#     record: Record,
#     axis: tp.Literal[0, 1] = 0,
#     index_name: tp.Any | None = None,
# ):
#     ...
#
# def append(
#     self,
#     record: Record,
#     axis: tp.Literal[0, 1] = 0,
#     index_name: tp.Any | None = None,
# ) -> None:
#     concat = Concat()
#     artifact = ArtifactFactory()([record])
#     table = TableFactory()([record])
#     a = record.apply(TableFactory())
#     if _is_artifact(self):
#         self.data = self.apply(concat, [artifact]).data
#     elif _is_experiment(self):
#         others = [
#             type(self)((artifact, table), self.name, self.autosave, self.savedir)
#         ]
#         self.data = self.apply(
#             concat, others, axis=axis, index_name=index_name
#         ).data
#     elif _is_record(self):
#         self.data = self.apply(concat, [record]).data
#     elif _is_table(self):
#         self.data = self.apply(
#             concat, [table], axis=axis, index_name=index_name
#         ).data
#     else:
#         raise TypeError(f"{self.__class__.__name__} does not support 'append'.")
#     self.operator = concat


@dataclass
class Record(Mapping):
    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))
    name: str | None = None

    def append(self, record: Record) -> None:
        concat: Concat[Record] = Concat()
        node = self.apply(concat, [record])
        if _is_record(node):
            self.data = node.data
            self.operator = node.operator
        else:
            raise TypeError(f"{self.__class__.__name__} does not support 'append'.")


@dataclass
class Artifact(Mapping):
    data: dict = field(default_factory=dict)


    def append(self, record: Record) -> None:
        other = ArtifactFactory()([record])
        node = self.apply(Concat(), [other])
        if _is_artifact(node):
            self.data = node.data
            self.operator = node.operator
        else:
            raise TypeError(f"{self.__class__.__name__} does not support 'append'.")


@dataclass
class Table(Mapping):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    name: str | None = None

    def append(self, record: Record) -> None:
        other = TableFactory()([record])
        node = self.apply(Concat(), [other], axis=0)
        if _is_table(node):
            self.data = node.data
            self.operator = node.operator
        else:
            raise TypeError(f"{self.__class__.__name__} does not support 'append'.")
