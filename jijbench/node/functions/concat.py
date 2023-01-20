from __future__ import annotations

import pandas as pd
import typing as tp

from jijbench.node.base import FunctionNode
import jijbench.node.data.artifact as _artifact
import jijbench.node.data.table as _table


if tp.TYPE_CHECKING:
    from jijbench.node.data import DataBase


class Concat(FunctionNode[DataBase, DataBase]):
    def __call__(
        self,
        inputs: list[DataBase],
        name: str | None = None,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> DataBase:
        dtype = type(inputs[0])
        if not all([isinstance(node, dtype) for node in inputs]):
            raise TypeError(
                "Type of elements of 'inputs' must be unified with either 'Table' or 'Artifact'."
            )

        if isinstance(inputs[0], _artifact.Artifact):
            data = inputs[0].data.copy()
            for node in inputs[1:]:
                if node.name in data:
                    data[node.name].update(node.data.copy())
                else:
                    data[node.name] = node.data.copy()
            return _artifact.Artifact(data=data, name=name)
        elif isinstance(inputs[0], _table.Table):
            data = pd.concat([node.data for node in inputs], axis=axis)
            data.index.name = index_name
            return _table.Table(data=data, name=name)
        else:
            raise TypeError(f"'{inputs[0].__class__.__name__}' type is not supported.")

    @property
    def name(self) -> str:
        return "concat"
