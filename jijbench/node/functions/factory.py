from __future__ import annotations

import numpy as np
import pandas as pd
import jijmodeling as jm
import typing as tp
import warnings

from jijbench.node.base import FunctionNode
import jijbench.node.data.artifact as _artifact
import jijbench.node.data.array as _array
import jijbench.node.data.record as _record
import jijbench.node.data.table as _table
import jijbench.node.data.value as _value

if tp.TYPE_CHECKING:
    from jijbench.node.base import DataNode



class ArtifactFactory(FunctionNode[_record.Record, _artifact.Artifact]):
    def __call__(self, inputs: list[_record.Record], name: str | None = None) -> _artifact.Artifact:
        data = {node.name: node.data.to_dict() for node in inputs}
        return _artifact.Artifact(data, name=name)

    @property
    def name(self) -> str:
        return "artifact"


class RecordFactory(FunctionNode[DataNode, _record.Record]):
    def __call__(
        self, inputs: list[DataNode], name: str | None = None, extract: bool = True
    ) -> _record.Record:
        data = {}
        for node in inputs:
            if isinstance(node.data, jm.SampleSet) and extract:
                data.update(
                    {n.name: n.data for n in self._to_nodes_from_sampleset(node.data)}
                )
            else:
                data[node.name] = node.data
        data = pd.Series(data)
        return _record.Record(data, name=name)

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
            _value.Value(sum(sampleset.feasible().record.num_occurrences), "num_feasible")
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


class TableFactory(FunctionNode[_record.Record, _table.Table]):
    def __call__(
        self,
        inputs: list[_record.Record],
        name: str | None = None,
        index_name: str | None = None,
    ) -> _table.Table:
        data = pd.DataFrame({node.name: node.data for node in inputs}).T
        data.index.name = index_name
        return _table.Table(data, name=name)

    @property
    def name(self) -> str:
        return "table"
