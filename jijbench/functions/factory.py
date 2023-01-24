from __future__ import annotations

import abc
import numpy as np
import pandas as pd
import jijmodeling as jm
import typing as tp
import warnings

from jijbench.node.base import FunctionNode, DataNodeKT_co, DataNodeVT_co
from jijbench.node.base import DataNode
from jijbench.data.mapping import Artifact, Table
from jijbench.data.record import Record
from jijbench.data.elements.array import Array
from jijbench.data.elements.value import Value


class Factory(FunctionNode[DataNodeKT_co, DataNodeVT_co]):
    def __call__(
        self, inputs: list[DataNodeKT_co], name: str | None = None, **kwargs: tp.Any
    ) -> DataNodeVT_co:
        return self.create(inputs, name, **kwargs)

    @abc.abstractmethod
    def create(
        self, inputs: list[DataNodeKT_co], name: str | None = None
    ) -> DataNodeVT_co:
        pass


class RecordFactory(Factory[DataNode, Record]):
    @property
    def name(self) -> str:
        return "record"

    def create(
        self,
        inputs: list[DataNode],
        name: str | None = None,
        is_parsed_sampleset: bool = True,
    ) -> Record:
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
        data = []

        data.append(
            Array(np.array(sampleset.record.num_occurrences), "num_occurrences")
        )
        data.append(Array(np.array(sampleset.evaluation.energy), "energy"))
        data.append(Array(np.array(sampleset.evaluation.objective), "objective"))

        constraint_violations = sampleset.evaluation.constraint_violations
        if constraint_violations:
            for k, v in constraint_violations.items():
                data.append(Array(np.array(v), k))

        data.append(Value(sum(sampleset.record.num_occurrences), "num_samples"))
        data.append(
            Value(sum(sampleset.feasible().record.num_occurrences), "num_feasible")
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
        data.append(Value(execution_time, "execution_time"))
        return data


class ArtifactFactory(Factory[Record, Artifact]):
    @property
    def name(self) -> str:
        return "artifact"

    def create(self, inputs: list[Record], name: str | None = None) -> Artifact:
        data = {node.name: node.data.to_dict() for node in inputs}
        return Artifact(data, name)


class TableFactory(Factory[Record, Table]):
    @property
    def name(self) -> str:
        return "table"

    def create(
        self,
        inputs: list[Record],
        name: str | None = None,
        index_name: str | None = None,
    ) -> Table:
        data = pd.DataFrame({node.name: node.data for node in inputs}).T
        data.index.name = index_name
        return Table(data, name)
