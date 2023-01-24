from __future__ import annotations

import pandas as pd
import typing as tp

from jijbench.node.base import FunctionNode
from jijbench.data.mapping import Artifact, Mapping, Table
from typing_extensions import TypeGuard


def _is_artifact_list(
    inputs: list[Artifact] | list[Table],
) -> TypeGuard[list[Artifact]]:
    return all([isinstance(node, Artifact) for node in inputs])


def _is_table_list(inputs: list[Artifact] | list[Table]) -> TypeGuard[list[Table]]:
    return all([isinstance(node, Table) for node in inputs])


class Concat(FunctionNode[Mapping, Mapping]):
    @tp.overload
    def operate(
        self,
        inputs: list[Artifact],
        name: str = "",
    ) -> Artifact:
        ...

    @tp.overload
    def operate(
        self,
        inputs: list[Table],
        name: str = "",
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> Table:
        ...

    def operate(
        self,
        inputs: list[Artifact] | list[Table],
        name: str = "",
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> Artifact | Table:
        if _is_artifact_list(inputs):
            data = {}
            for node in inputs:
                data.update(node.data.copy())
            return Artifact(data, name)
        elif _is_table_list(inputs):
            data = pd.concat([node.data for node in inputs], axis=axis)
            data.index.name = index_name
            return Table(data, name)
        else:
            raise TypeError(
                "Type of elements of 'inputs' must be unified with either 'Table' or 'Artifact'."
            )
