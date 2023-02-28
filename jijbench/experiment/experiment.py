from __future__ import annotations

import abc
import pandas as pd
import pathlib
import uuid
import warnings

from dataclasses import dataclass, field
from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.elements.base import Callable
from jijbench.elements.id import ID
from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, RecordFactory, TableFactory
from jijbench.io.io import save
from jijbench.mappings.mappings import Artifact, Mapping, Record, Table
from jijbench.solver.base import Parameter, Response
from jijbench.typing import ExperimentDataType


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

        setattr(self, "state", _Created())

    def __len__(self) -> int:
        """
        Perform the operation __len__.
        """
        a_len = len(self.data[0])
        t_len = len(self.data[1])
        if a_len != t_len:
            warnings.warn(
                f"The length of artifact object and table object are different: {a_len} != {t_len}, return the larger one."
            )
        return max(a_len, t_len)

    def __enter__(self) -> Experiment:
        """Set up Experiment.
        Automatically makes a directory for saving the experiment, if it doesn't exist."""

        setattr(self, "state", _Running())

        savedir = (
            self.savedir
            if isinstance(self.savedir, pathlib.Path)
            else pathlib.Path(self.savedir)
        )
        savedir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Saves the experiment if autosave is True."""
        state = getattr(self, "state")

        if self.autosave:
            state.save(self)

        setattr(self, "state", _Waiting())

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
        bools = self.data[1].data.applymap(lambda x: isinstance(x, Response))
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
        state = getattr(self, "state")
        state.append(self, record)

    def save(self):
        """Save the experiment."""
        savedir = (
            self.savedir
            if isinstance(self.savedir, pathlib.Path)
            else pathlib.Path(self.savedir)
        )
        savedir.mkdir(parents=True, exist_ok=True)
        state = getattr(self, "state")
        state.save(self)


class _ExperimentState(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def append(self, context: Experiment, record: Record) -> None:
        pass

    @abc.abstractmethod
    def save(self) -> None:
        pass


class _Created(_ExperimentState):
    def append(self, context: Experiment, record: Record) -> None:
        record.name = len(context)
        _append(context, record)
        context.state = _Waiting()

    def save(self, context: Experiment) -> None:
        save(context, savedir=context.savedir)


class _Waiting(_ExperimentState):
    def append(self, context: Experiment, record: Record) -> None:
        record.name = len(context)
        _append(context, record)

    def save(self, context) -> None:
        save(context, savedir=context.savedir)


class _Running(_ExperimentState):
    def append(self, context: Experiment, record: Record) -> None:
        experiment_id, run_id = ID(name="experiment_id"), ID(name="run_id")
        ids = RecordFactory()([experiment_id, run_id])
        record.name = tuple(ids.view().tolist())
        _append(context, record)
        _, table = context.data
        table.index.names = [experiment_id.name, run_id.name]

    def save(
        self,
        context: Experiment,
    ) -> None:
        state: _Running = getattr(context, "state")

        a, t = context.data
        ids = tuple(t.index)
        print(tuple(t.index))
        print(a.keys())
        fafaf

        ai = Artifact({ids: a.data[ids]}, a.name)
        ti = Table(t.data.loc[[ids]], t.name)

        experiment = Experiment(
            (ai, ti), context.name, autosave=context.autosave, savedir=context.savedir
        )
        save(experiment, savedir=context.savedir, mode="a")


def _append(context: Experiment, record: Record) -> None:
    concat: Concat[Experiment] = Concat()
    data = (ArtifactFactory()([record]), TableFactory()([record]))
    other = type(context)(
        data, context.name, autosave=context.autosave, savedir=context.savedir
    )
    node = context.apply(
        concat,
        [other],
        name=context.name,
        autosave=context.autosave,
        savedir=context.savedir,
    )
    context._init_attrs(node)
