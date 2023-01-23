from __future__ import annotations

import numpy as np
import pandas as pd
import jijmodeling as jm
import warnings

from jijbench.node.base import FunctionNode

from jijbench.node.base import DataNode
from jijbench.node.data.database import Artifact, Table
from jijbench.node.data.array import Array
from jijbench.node.data.record import Record
from jijbench.node.data.value import Value


class ArtifactFactory(FunctionNode[Record, Artifact]):
    def __call__(self, inputs: list[Record], name: str | None = None) -> Artifact:
        data = {node.name: node.data.to_dict() for node in inputs}
        return Artifact(data, name=name)

    @property
    def name(self) -> str:
        return "artifact"


class RecordFactory(FunctionNode[DataNode, Record]):
    def __call__(
        self, inputs: list[DataNode], name: str | None = None, extract: bool = True
    ) -> Record:
        data = {}
        for node in inputs:
            if isinstance(node.data, jm.SampleSet) and extract:
                data.update(
                    {n.name: n.data for n in self._to_nodes_from_sampleset(node.data)}
                )
            else:
                data[node.name] = node.data
        data = pd.Series(data)
        return Record(data, name=name)

    @property
    def name(self) -> str:
        return "record"

    def _to_nodes_from_sampleset(self, sampleset: jm.SampleSet) -> list[DataNode]:
        data = []

        data.append(
            _array.Array(np.array(sampleset.record.num_occurrences), "num_occurrences")
        )
        data.append(_array.Array(np.array(sampleset.evaluation.energy), "energy"))
        data.append(_array.Array(np.array(sampleset.evaluation.objective), "objective"))

        constraint_violations = sampleset.evaluation.constraint_violations
        if constraint_violations:
            for k, v in constraint_violations.items():
                data.append(_array.Array(np.array(v), k))

        data.append(_value.Value(sum(sampleset.record.num_occurrences), "num_samples"))
        data.append(
            _value.Value(
                sum(sampleset.feasible().record.num_occurrences), "num_feasible"
            )
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
        data.append(_value.Value(execution_time, "execution_time"))
        return data


class TableFactory(FunctionNode[Record, Table]):
    def __call__(
        self,
        inputs: list[Record],
        name: str | None = None,
        index_name: str | None = None,
    ) -> Table:
        data = pd.DataFrame({node.name: node.data for node in inputs}).T
        data.index.name = index_name
        return Table(data, name=name)

    @property
    def name(self) -> str:
        return "table"
