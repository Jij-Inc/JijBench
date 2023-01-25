from __future__ import annotations

import pandas as pd
import typing as tp

from dataclasses import dataclass, field
from jijbench.node.base import DataNode

if tp.TYPE_CHECKING:
    from jijbench.experiment.experiment import Experiment


@dataclass
class Mapping(DataNode):
    @tp.overload
    def append(self: Artifact, record: Record) -> None:
        ...

    @tp.overload
    def append(
        self: Table,
        record: Record,
        axis: tp.Literal[0, 1] = 0,
        index_name: tp.Any | None = None,
    ) -> None:
        ...

    @tp.overload
    def append(
        self: Experiment,
        record,
    ):
        ...

    def append(
        self,
        record: Record,
        axis: tp.Literal[0, 1] = 0,
        index_name: tp.Any | None = None,
    ) -> None:
        from jijbench.experiment.experiment import Experiment
        from jijbench.functions.concat import Concat
        from jijbench.functions.factory import ArtifactFactory, TableFactory

        concat = Concat()
        if isinstance(self, Artifact):
            node = record.apply(ArtifactFactory(), name=self.name)
            self.data = self.apply(
                concat, [node], axis=axis, index_name=index_name
            ).data
        elif isinstance(self, Record):
            self.data = self.apply(concat, [record]).data
        elif isinstance(self, Table):
            node = record.apply(TableFactory(), name=self.name)
            self.data = self.apply(
                concat, [node], axis=axis, index_name=index_name
            ).data
        elif isinstance(self, Experiment):
            for d in self.data:
                if isinstance(d, Artifact):
                    d.append(record)
                elif isinstance(d, Table):
                    d.append(record, index_name=("experiment_id", "run_id"))
        else:
            raise TypeError(f"{self.__class__.__name__} does not support 'append'.")

        self.operator = concat


@dataclass
class Record(Mapping):
    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))
    name: str = ""


@dataclass
class Artifact(Mapping):
    data: dict = field(default_factory=dict)


@dataclass
class Table(Mapping):
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
