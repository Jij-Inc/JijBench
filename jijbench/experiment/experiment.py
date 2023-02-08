from __future__ import annotations

import pandas as pd
import typing as tp
import pathlib
import uuid

from dataclasses import dataclass, field
from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.elements.base import Callable
from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, TableFactory
from jijbench.io.io import save
from jijbench.mappings.mappings import Artifact, Mapping, Table
from jijbench.solver.solver import Parameter, Return
from jijbench.typing import ExperimentDataType


if tp.TYPE_CHECKING:
    from jijbench.mappings.mappings import Record


@dataclass
class Experiment(Mapping[ExperimentDataType]):
    data: tuple[Artifact, Table] = field(default_factory=lambda: (Artifact(), Table()))
    name: str = field(default_factory=lambda: str(uuid.uuid4()))
    autosave: bool = field(default=True, repr=False)
    savedir: str | pathlib.Path = field(default=DEFAULT_RESULT_DIR, repr=False)

    def __post_init__(self):
        if self.data[0].name is None:
            self.data[0].name = self.name

        if self.data[1].name is None:
            self.data[1].name = self.name

        if isinstance(self.savedir, str):
            self.savedir = pathlib.Path(self.savedir)

    def __enter__(self) -> Experiment:
        savedir = (
            self.savedir
            if isinstance(self.savedir, pathlib.Path)
            else pathlib.Path(self.savedir)
        )
        savedir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
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

    @classmethod
    def validate_data(cls, data: ExperimentDataType) -> ExperimentDataType:
        artifact, table = data
        if not isinstance(artifact, Artifact):
            raise TypeError(
                f"Type of attribute data is {ExperimentDataType}, and data[0] must be Artifact instead of {type(artifact).__name__}."
            )
        if not isinstance(table, Table):
            raise TypeError(
                f"Type of attribute data is {ExperimentDataType}, and data[1] must be Table instead of {type(artifact).__name__}."
            )
        return data

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
        save(self, savedir=self.savedir, mode="a")
