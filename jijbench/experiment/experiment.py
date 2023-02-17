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
from jijbench.solver.base import Parameter, Return
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
        """Makes a directory for saving the experiment, if it doesn't exist. Returns the experiment object."""
        savedir = (
            self.savedir
            if isinstance(self.savedir, pathlib.Path)
            else pathlib.Path(self.savedir)
        )
        savedir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Saves the experiment if autosave is True."""
        if self.autosave:
            self.save()

    @property
    def artifact(self) -> dict:
        """Return the artifact of the experiment as a dictionary."""
        return self.data[0].view()

    @property
    def table(self) -> pd.DataFrame:
        """Return the table of the experiment as a pandas dataframe."""
        return self.data[1].view()

    @property
    def params_table(self) -> pd.DataFrame:
        """Return the parameters table of the experiment as a pandas dataframe."""
        bools = self.data[1].data.applymap(lambda x: isinstance(x, Parameter))
        return self.table[bools].dropna(axis=1)

    @property
    def solver_table(self) -> pd.DataFrame:
        """Return the solver table of the experiment as a pandas dataframe."""

        bools = self.data[1].data.applymap(lambda x: isinstance(x, Callable))
        return self.table[bools].dropna(axis=1)

    @property
    def returns_table(self) -> pd.DataFrame:
        """Return the returns table of the experiment as a pandas dataframe."""
        bools = self.data[1].data.applymap(lambda x: isinstance(x, Return))
        return self.table[bools].dropna(axis=1)

    @classmethod
    def validate_data(cls, data: ExperimentDataType) -> ExperimentDataType:
        """Validate the data of the experiment.

        Args:
            data (ExperimentDataType): The data to validate.

        Raises:
            TypeError: If data is not an instance of ExperimentDataType or
            if the first element of data is not an instance of Artifact or
            if the second element of data is not an instance of Table.

        Returns:
            ExperimentDataType: The validated data.
        """
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

    def view(self) -> tuple[dict, pd.DataFrame]:
        """Return a tuple of the artifact dictionary and table dataframe."""
        return (self.data[0].view(), self.data[1].view())

    def append(self, record: Record) -> None:
        """Append a new record to the experiment.

        Args:
            record (Record): The record to be appended to the experiment.
        """
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
        self._init_attrs(node)

    def save(self):
        """Save the experiment."""
        save(self, savedir=self.savedir, mode="a")
