from __future__ import annotations

import dill
import pandas as pd
import typing as tp
import pathlib

from dataclasses import dataclass
from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.data.mapping import Artifact, Mapping, Table
from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, TableFactory
from jijbench.data.elements.id import ID
from typing_extensions import TypeGuard


if tp.TYPE_CHECKING:
    from jijbench.data.mapping import Record


def _is_experiment(
    node: Mapping,
) -> TypeGuard[Experiment]:
    return node.__class__.__name__ == "Experiment"


@dataclass
class Experiment(Mapping):
    def __init__(
        self,
        data: tuple[Artifact, Table] | None = None,
        name: str | None = None,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ):
        if name is None:
            name = ID().data

        if data is None:
            data = (Artifact(), Table())

        if data[0].name is None:
            data[0].name = name

        if data[1].name is None:
            data[1].name = name

        self.data = data
        self.name = name
        self.autosave = autosave

        if isinstance(savedir, str):
            savedir = pathlib.Path(savedir)
        self.savedir = savedir

    def __enter__(self) -> Experiment:
        p = tp.cast("pathlib.Path", self.savedir) / str(self.name)
        (p / "table").mkdir(parents=True, exist_ok=True)
        (p / "artifact").mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        index = (self.name, self.table.index[-1])
        self.table.rename(index={self.table.index[-1]: index}, inplace=True)

        if self.autosave:
            self.save()

    @property
    def artifact(self) -> dict:
        return self.data[0].data

    @property
    def table(self) -> pd.DataFrame:
        t = self.data[1].data
        is_tuple_index = all([isinstance(i, tuple) for i in t.index])
        if is_tuple_index:
            names = t.index.names if len(t.index.names) >= 2 else None
            index = pd.MultiIndex.from_tuples(t.index, names=names)
            t.index = index
        return t

    def append(self, record: Record) -> None:
        data = (ArtifactFactory()([record]), TableFactory()([record]))
        other = type(self)(data, self.name, self.autosave, self.savedir)
        node = self.apply(Concat(), [other])
        if _is_experiment(node):
            self.data = node.data
            self.operator = node.operator
        else:
            raise TypeError(f"{self.__class__.__name__} does not support 'append'.")

    def append(self, record: Record) -> None:
        for d in self.data:
            d.append(record, index_name=("experiment_id", "run_id"))

    def concat(self, experiment: Experiment) -> None:
        from jijbench.functions.concat import Concat

        c = Concat()

        artifact = c([self.data[0], experiment.data[0]])
        table = c([self.data[1], experiment.data[1]])

        self.data = (artifact, table)
        self.operator = c

    def save(self):
        def is_dillable(obj: tp.Any):
            try:
                dill.dumps(obj)
                return True
            except Exception:
                return False

        p = self.savedir / str(self.name) / "table" / "table.csv"
        self.table.to_csv(p)

        p = self.savedir / str(self.name) / "artifact" / "artifact.dill"
        record_name = list(self.data[0].operator.inputs[1].data.keys())[0]
        if p.exists():
            with open(p, "rb") as f:
                artifact = dill.load(f)
                artifact[self.name][record_name] = {}
        else:
            artifact = {self.name: {record_name: {}}}

        record = {}
        for k, v in self.artifact[self.name][record_name].items():
            if is_dillable(v):
                record[k] = v
            else:
                record[k] = str(v)
        artifact[self.name][record_name].update(record)

        with open(p, "wb") as f:
            dill.dump(artifact, f)
