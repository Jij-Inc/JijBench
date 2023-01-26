from __future__ import annotations

import dill
import pandas as pd
import typing as tp
import pathlib

from dataclasses import dataclass, field
from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.data.mapping import Artifact, Mapping, Table
from jijbench.data.elements.id import ID


@dataclass
class Experiment(Mapping):
    data: tuple[Artifact, Table] = field(default_factory=lambda: (Artifact(), Table()))
    name: str | None = None
    autosave: bool = True
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR

    def __post_init__(self):
        if self.name is None:
            self.name = ID().data

        if self.data[0].name is None:
            self.data[0].name = self.name

        if self.data[1].name is None:
            self.data[1].name = self.name

        if isinstance(self.savedir, str):
            self.savedir = pathlib.Path(self.savedir)

    @property
    def artifact(self) -> dict:
        return self.data[0].data

    @property
    def table(self) -> pd.DataFrame:
        t = self.data[1].data
        if t.empty:
            return t
        else:
            is_tuple_index = all([isinstance(i, tuple) for i in t.index])
            if is_tuple_index:
                names = t.index.names if len(t.index.names) >= 2 else None
                index = pd.MultiIndex.from_tuples(t.index, names=names)
                t.index = index
            return t

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

    # def concat(self, experiment: Experiment) -> None:
    #    from jijbench.functions.concat import Concat
    #
    #    concat = Concat()
    #
    #    artifact = concat([self.data[0], experiment.data[0]])
    #    table = concat([self.data[1], experiment.data[1]])
    #
    #    self.data = (artifact, table)
    #    self.operator = concat

    def save(self):
        def is_dillable(obj: tp.Any):
            try:
                dill.dumps(obj)
                return True
            except Exception:
                return False

        savedir = tp.cast("pathlib.Path", self.savedir)
        p = savedir / str(self.name) / "table" / "table.csv"
        self.table.to_csv(p)

        p = savedir / str(self.name) / "artifact" / "artifact.dill"
        record_name = list(self.data[0].operator.inputs[1].data.keys())[0]
        if p.exists():
            with open(p, "rb") as f:
                artifact = dill.load(f)
                artifact[self.name][record_name] = {}
        else:
            artifact = {self.name: {record_name: {}}}

        record = {}
        for k, v in self.artifact[record_name].items():
            if is_dillable(v):
                record[k] = v
            else:
                record[k] = str(v)
        artifact[self.name][record_name].update(record)

        with open(p, "wb") as f:
            dill.dump(artifact, f)
