from __future__ import annotations

import abc
import numpy as np
import pandas as pd
import jijmodeling as jm
import typing as tp
import warnings

from dataclasses import dataclass, field
from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, TableFactory
from jijbench.node.base import DataNode
from jijbench.typing import T, ArtifactDataType


@dataclass
class Container(DataNode[T]):
    """An abstract class for all Container classes that implements the methods to be
    followed by all child classes.
    """

    @abc.abstractmethod
    def __len__(self) -> int:
        """
        Perform the operation __len__.
        """
        pass

    @abc.abstractmethod
    def append(self, record: Record) -> None:
        """Append method to be implemented in the child classes.

        Args:
            record: the record to be appended.

        Returns:
            A data type of class T.
        """
        pass

    @abc.abstractmethod
    def view(self) -> T:
        """View method to be implemented in the child classes."""
        pass


@dataclass
class Record(Container[pd.Series]):
    """Data structure that maps data onto a `pandas.Series`.

    Attributes:
        data (pandas.Series): pandas series object.
        name (Hashable): The name of the record. Defaults to None.
    """

    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))
    name: tp.Hashable = None

    def __len__(self) -> int:
        """
        Perform the operation __len__.
        """
        return len(self.data)

    @classmethod
    def validate_data(cls, data: pd.Series) -> pd.Series:
        """
        Validate the data to ensure that it is a pandas Series and all elements of the Series are instances of DataNode.

        Args:
            data (pandas.Series): The data to be validated.

        Raises:
            TypeError: If the input data is not a pandas Series or if not all elements of the Series are instances of DataNode.

        Returns:
            The validated data.

        """
        data = cls._validate_dtype(data, (pd.Series,))
        if data.empty:
            return data
        else:
            if data.apply(lambda x: isinstance(x, DataNode)).all():
                return data
            else:
                raise TypeError(
                    f"All elements of {data.__class__.__name__} must be type DataNode."
                )

    @property
    def index(self) -> pd.Index:
        """Return the index of the Record data."""
        return self.data.index

    @index.setter
    def index(self, index: pd.Index) -> None:
        """Set the index of the Record data.

        Args:
            index (pd.Index): The index to set for the Record data.
        """
        self.data.index = index

    def append(self, record: Record) -> None:
        """Apeend a new Record to the current Record.

        Args:
            record (Record): The Record to be added.
        """
        concat: Concat[Record] = Concat()
        node = self.apply(concat, [record], name=self.name)
        self._init_attrs(node)

    def view(self) -> pd.Series:
        """Return the data of each DataNode in the Series as a new Series."""
        return self.data.apply(lambda x: x.data)


@dataclass
class Artifact(Container[ArtifactDataType]):
    """Data structure that maps data onto a `dict`.

    Attributes:
        data (ArtifactDataType): The data stored in the Artifact.
        name (Hashable): The name of the Artifact. Defaults to None.
    """

    data: ArtifactDataType = field(default_factory=dict)
    name: tp.Hashable = None

    def __len__(self) -> int:
        """
        Perform the operation __len__.
        """
        return len(self.data)

    @classmethod
    def validate_data(cls, data: ArtifactDataType) -> ArtifactDataType:
        """Validate the data stored in the Artifact.

        The data in the Artifact must be of type `dict`. The values stored in
        the `dict` must be of type `DataNode`.

        Args:
            data (ArtifactDataType): The data to be validated.

        Raises:
            TypeError: If the data is not of the correct type.

        Returns:
            ArtifactDataType: The validated data.
        """
        if data:
            data = cls._validate_dtype(data, (dict,))
            values = []
            for v in data.values():
                if isinstance(v, dict):
                    values += list(v.values())
                else:
                    raise TypeError(
                        f"Type of attibute data is {ArtifactDataType}. Input data is invaid."
                    )
            if all(map(lambda x: isinstance(x, DataNode), values)):
                return data
            else:
                raise TypeError(
                    f"Type of attibute data is {ArtifactDataType}. Input data is invaid."
                )
        else:
            return data

    def keys(self) -> tuple[tp.Hashable, ...]:
        """Return a tuple of keys."""
        return tuple(self.data.keys())

    def values(self) -> tuple[dict[tp.Hashable, DataNode], ...]:
        """Return a tuple of values."""
        return tuple(self.data.values())

    def items(
        self,
    ) -> tuple[tuple[tp.Hashable, dict[tp.Hashable, DataNode]], ...]:
        """Return a tuple of key-value pairs."""
        return tuple(self.data.items())

    def append(self, record: Record) -> None:
        """Append a new record to the Artifact.

        Args:
            record (Record): The data to be appended.
        """
        concat: Concat[Artifact] = Concat()
        other = ArtifactFactory()([record])
        node = self.apply(concat, [other], name=self.name)
        self._init_attrs(node)

    def view(self) -> ArtifactDataType:
        """Return the data of each DataNode in the dict as a new dict."""
        return {
            k: {name: node.data for name, node in v.items()}
            for k, v in self.data.items()
        }


@dataclass
class Table(Container[pd.DataFrame]):
    """Data structure that maps data onto a `pandas.DataFrame`.

    This class is one of Container. The element in the each cell is a DataNode.
    Table class extends the basic functionality of a `pandas.DataFrame` with the ability to store and manipulate `DataNode` objects.

    Attributes:
        data (pd.DataFrame): The actual data stored in the table.
        name (tp.Hashable): The name of the Table, which is used as an identifier.
    """

    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    name: tp.Hashable = None

    def __len__(self) -> int:
        """
        Perform the operation __len__.
        """
        return len(self.data)

    @classmethod
    def validate_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        """Validate the data to ensure it is a pandas DataFrame with all elements being of type `DataNode`.

        Args:
            data (pandas.DataFrame): The data to validate.

        Raises:
            TypeError: If the input data is not a pandas DataFrame or if it contains elements that are not of type `DataNode`.

        Returns:
            pandas.DataFrame: The validated data.
        """
        data = cls._validate_dtype(data, (pd.DataFrame,))
        if data.empty:
            return data
        else:
            if data.applymap(lambda x: isinstance(x, DataNode)).values.all():
                return data
            else:
                raise TypeError(
                    f"All elements of {data.__class__.__name__} must be type DataNode."
                )

    @property
    def index(self) -> pd.Index:
        """Return the index of the data in the Table object."""
        return self.data.index

    @index.setter
    def index(self, index: pd.Index) -> None:
        """Set the index of the data in the Table object.

        Args:
            index (pandas.Index): The new index for the data stored in the Table object.
        """
        self.data.index = index

    @property
    def columns(self) -> pd.Index:
        """Return the columns of the data in the Table object."""
        return self.data.columns

    @columns.setter
    def columns(self, columns: pd.Index) -> None:
        """Set the columns of the data in the Table object.

        Args:
            columns (pandas.Index): The new columns for the data stored in the Table object.
        """
        self.data.columns = columns

    def append(self, record: Record) -> None:
        """Append a new record to the Table.

        Args:
            record (Record): The data to be appended.
        """
        concat: Concat[Table] = Concat()
        other = TableFactory()([record])
        node = self.apply(concat, [other], name=self.name, axis=0)
        self._init_attrs(node)

    def view(self) -> pd.DataFrame:
        """Return the data of each DataNode in the pandas.DataFrame as a new pandas.DataFrame."""
        if self.data.empty:
            return self.data
        else:
            data = self.data.applymap(lambda x: x.data)
            sampleset_columns = [
                c for c in data.columns if isinstance(data[c][0], jm.SampleSet)
            ]
            if sampleset_columns:
                extracted = pd.concat(
                    [data[c].apply(self._extract) for c in sampleset_columns], axis=1
                )
                data.drop(sampleset_columns, axis=1, inplace=True)
                return pd.concat([data, extracted], axis=1)
            else:
                return data

    def _extract(self, sampleset: jm.SampleSet) -> pd.Series:
        """Extract data from jijmodeling.SampleSet object.

        This method extracts relevant data from a `jijmodeling.SampleSet`, such as the number of occurrences,
        energy, objective, constraint violations, number of samples, number of feasible samples, and the
        execution time. The extracted data is returned as a list of DataNode objects.


        Args:
            node (jijmodeling.SampleSet): A jijmodeling SampleSet from which to extract data.

        Returns:
            pd.Series: Extracted data from the SampleSet..
        """

        data = {}
        data["num_occurrences"] = np.array(sampleset.record.num_occurrences)
        data["energy"] = np.array(sampleset.evaluation.energy)
        data["objective"] = np.array(sampleset.evaluation.objective)

        constraint_violations = sampleset.evaluation.constraint_violations
        if constraint_violations:
            for k, v in constraint_violations.items():
                data[f"{k}_violations"] = np.array(v)

        data["num_samples"] = sum(sampleset.record.num_occurrences)
        data["num_feasible"] = sum(sampleset.feasible().record.num_occurrences)

        # TODO スキーマが変わったら修正
        solving_time = sampleset.measuring_time.solve
        if solving_time is None:
            execution_time = np.nan
            warnings.warn(
                "'solve' of jijmodeling.SampleSet is None. Give it if you want to evaluate automatically."
            )
        else:
            if solving_time.solve is None:
                execution_time = np.nan
                warnings.warn(
                    "'solve' of jijmodeling.SampleSet is None. Give it if you want to evaluate automatically."
                )
            else:
                execution_time = solving_time.solve
        data["execution_time"] = execution_time
        return pd.Series(data)
