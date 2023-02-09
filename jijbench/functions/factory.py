from __future__ import annotations

import abc
import numpy as np
import pandas as pd
import jijmodeling as jm
import typing as tp
import warnings

from jijbench.node.base import DataNode, FunctionNode
from jijbench.elements.array import Array
from jijbench.elements.base import Number
from jijbench.typing import DataNodeT, DataNodeT2

if tp.TYPE_CHECKING:
    from jijbench.mappings.mappings import Artifact, Record, Table


class Factory(FunctionNode[DataNodeT, DataNodeT2]):
    """An abstract base class for creating a new data node from a list of input nodes.

    Attributes:
        inputs (List[DataNodeT]): List of input data nodes.
        name (str or None): Name of the resulting data node.

    """

    @abc.abstractmethod
    def create(self, inputs: list[DataNodeT], name: str | None = None) -> DataNodeT2:
        """Abstract method to create a new data node.
        Subclasses must implement this method.

        Args:
            inputs (List[DataNodeT]): List of input data nodes.
            name (str or None): Name of the resulting data node.

        Returns:
            DataNodeT2: The resulting data node.

        """
        pass

    def operate(
        self, inputs: list[DataNodeT], name: str | None = None, **kwargs: tp.Any
    ) -> DataNodeT2:
        """Create a new data node from the given inputs.
        This method calls `create` method to create a new data node from the given inputs.

        Args:
            inputs (List[DataNodeT]): List of input data nodes.
            name (str or None, optional): Name of the resulting data node.
            **kwargs: Additional keyword arguments.

        Returns:
            DataNodeT2: The resulting data node.

        """
        return self.create(inputs, name, **kwargs)


class RecordFactory(Factory[DataNodeT, "Record"]):
    """A factory class for creating Record objects.

    This class creates Record objects from a list of input DataNode objects. It uses the `create` method to
    process the input DataNodes, extract their data and convert it into a pandas Series. The resulting Series is
    used to create the Record object. The class also includes a helper method `_to_nodes_from_sampleset` which
    is used to extract the relevant data from `jijmodeling.SampleSet` objects.
    """

    def create(
        self,
        inputs: list[DataNodeT],
        name: str = "",
        is_parsed_sampleset: bool = True,
    ) -> Record:
        """Create a Record object from the input DataNode objects.

        This method takes a list of input DataNode objects, processes them and converts them into a pandas
        Series. The resulting Series is used to create the Record object. If the input data is a `jijmodeling.SampleSet`
        and `is_parsed_sampleset` is set to True, the relevant data is extracted from the SampleSet using the
        `_to_nodes_from_sampleset` helper method.

        Args:
            inputs (list[DataNodeT]): A list of DataNode objects to be processed.
            name (str, optional): A name for the Record object. Defaults to "".
            is_parsed_sampleset (bool, optional): Whether to extract data from jijmodeling SampleSet objects. Defaults to True.

        Returns:
            Record: A Record object created from the processed input DataNode objects.
        """
        from jijbench.mappings.mappings import Record

        data = {}
        for node in inputs:
            if isinstance(node.data, jm.SampleSet) and is_parsed_sampleset:
                data.update(
                    {n.name: n for n in self._to_nodes_from_sampleset(node.data)}
                )
            else:
                data[node.name] = node
        data = pd.Series(data)
        return Record(data, name)

    def _to_nodes_from_sampleset(self, sampleset: jm.SampleSet) -> list[DataNode]:
        """Extract relevant data from a jijmodeling SampleSet.

        This method extracts relevant data from a `jijmodeling.SampleSet`, such as the number of occurrences,
        energy, objective, constraint violations, number of samples, number of feasible samples, and the
        execution time. The extracted data is returned as a list of DataNode objects.

        Args:
            sampleset (jijmodeling.SampleSet): A jijmodeling SampleSet from which to extract data.

        Returns:
            list[DataNode]: A list of DataNode objects containing the extracted data from the SampleSet.
        """
        data = []

        data.append(
            Array(np.array(sampleset.record.num_occurrences), "num_occurrences")
        )
        data.append(Array(np.array(sampleset.evaluation.energy), "energy"))
        data.append(Array(np.array(sampleset.evaluation.objective), "objective"))

        constraint_violations = sampleset.evaluation.constraint_violations
        if constraint_violations:
            for k, v in constraint_violations.items():
                data.append(Array(np.array(v), f"{k}_violations"))

        data.append(Number(sum(sampleset.record.num_occurrences), "num_samples"))
        data.append(
            Number(sum(sampleset.feasible().record.num_occurrences), "num_feasible")
        )

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
        data.append(Number(execution_time, "execution_time"))
        return data


class ArtifactFactory(Factory["Record", "Artifact"]):
    """A factory class for creating Artifact objects."""

    def create(self, inputs: list[Record], name: str = "") -> Artifact:
        """Creates an `Artifact` object using a list of `Record` inputs.

        Args:
            inputs (list[Record]): A list of `Record` objects to be used to create the `Artifact`.
            name (str, optional): Name of the `Artifact` object. Defaults to an empty string.

        Returns:
            Artifact: The created `Artifact` object.
        """
        from jijbench.mappings.mappings import Artifact

        data = {
            node.name
            if isinstance(node.name, tp.Hashable)
            else str(node.name): node.data.to_dict()
            for node in inputs
        }
        return Artifact(data, name)


class TableFactory(Factory["Record", "Table"]):
    """A factory class for creating Table objects."""

    def create(
        self,
        inputs: list[Record],
        name: str = "",
        index_name: str | None = None,
    ) -> Table:
        """Creates a `Table` object using a list of `Record` inputs.

        Args:
            inputs (list[Record]): A list of `Record` objects to be used to create the `Table`.
            name (str, optional): Name of the `Table` object. Defaults to an empty string.
            index_name (str | None, optional): Name of the index in the created `Table`. Defaults to None.

        Returns:
            Table: The created `Table` object.
        """
        from jijbench.mappings.mappings import Table

        data = pd.DataFrame({node.name: node.data for node in inputs}).T
        data.index.name = index_name
        return Table(data, name)
