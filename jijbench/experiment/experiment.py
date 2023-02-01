from __future__ import annotations

import dill
import pandas as pd
import typing as tp
import pathlib

from dataclasses import dataclass, field
from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.data.mapping import Artifact, Mapping, Table
from jijbench.data.elements.values import Callable, Parameter, Return
from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, TableFactory
from jijbench.data.elements.id import ID


if tp.TYPE_CHECKING:
    from jijbench.data.mapping import Record


@dataclass
class Experiment(Mapping):
    data: tuple[Artifact, Table] = field(default_factory=lambda: (Artifact(), Table()))
    name: str | None = None
    autosave: bool = field(default=True, repr=False)
    savedir: str | pathlib.Path = field(default=DEFAULT_RESULT_DIR, repr=False)

    def __post_init__(self):
        if self.name is None:
            self.name = ID().data

        if self.data[0].name is None:
            self.data[0].name = self.name

        if self.data[1].name is None:
            self.data[1].name = self.name

        if isinstance(self.savedir, str):
            self.savedir = pathlib.Path(self.savedir)

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
        return self.view("artifact")

    @property
    def table(self) -> pd.DataFrame:
        return self.view("table")

    @property
    def params_table(self) -> pd.DataFrame:
        bools = self.data[1].data.applymap(lambda x: isinstance(x, Parameter))
        return self.table[bools].dropna(axis=1)

    @property
    def solver_table(self) -> pd.DataFrame:
        bools = self.data[1].data.applymap(lambda x: isinstance(x, Callable))
        return self.table[bools].dropna(axis=1)

    @property
    def returns_table(self) -> pd.DataFrame:
        bools = self.data[1].data.applymap(lambda x: isinstance(x, Return))
        return self.table[bools].dropna(axis=1)

    @tp.overload
    def view(self, kind: tp.Literal["artifact"]) -> dict:
        ...

    @tp.overload
    def view(self, kind: tp.Literal["table"]) -> pd.DataFrame:
        ...

    def view(self, kind: tp.Literal["artifact", "table"]) -> dict | pd.DataFrame:
        if kind == "artifact":
            return self.data[0].view()
        else:
            return self.data[1].view()

    def append(self, record: Record) -> None:
        concat: Concat[Experiment] = Concat()
        data = (ArtifactFactory()([record]), TableFactory()([record]))
        other = type(self)(
            data, self.name, autosave=self.autosave, savedir=self.savedir
        )
        node = self.apply(
            concat,
            [other],
            name=self.name,
            autosave=self.autosave,
            savedir=self.savedir,
        )
        self.__init__(**node.__dict__)

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
        record_name = list(self.artifact.keys())[-1]
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
