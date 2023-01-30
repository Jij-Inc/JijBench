from __future__ import annotations

import pandas as pd
import typing as tp

from jijbench.node.base import FunctionNode
from jijbench.typing import MappingT, MappingTypes, MappingListTypes
from typing_extensions import TypeGuard

if tp.TYPE_CHECKING:
    from jijbench.data.mapping import Artifact, Experiment, Record, Table


def _is_artifact_list(
    inputs: MappingListTypes,
) -> TypeGuard[list[Artifact]]:
    return all([node.__class__.__name__ == "Artifact" for node in inputs])


def _is_experiment_list(
    inputs: MappingListTypes,
) -> TypeGuard[list[Experiment]]:
    return all([node.__class__.__name__ == "Experiment" for node in inputs])


def _is_record_list(
    inputs: MappingListTypes,
) -> TypeGuard[list[Record]]:
    return all([node.__class__.__name__ == "Record" for node in inputs])


def _is_table_list(
    inputs: MappingListTypes,
) -> TypeGuard[list[Table]]:
    return all([node.__class__.__name__ == "Table" for node in inputs])


def _is_mapping_list(inputs: MappingListTypes) -> TypeGuard[list[MappingTypes]]:
    cls_name = inputs[0].__class__.__name__
    return all([node.__class__.__name__ == cls_name for node in inputs])


class Concat(FunctionNode[MappingTypes, MappingT]):
    # @tp.overload
    # def __call__(self, inputs: list[Artifact], name: str = "") -> Artifact:
    #     ...
    #
    # @tp.overload
    # def __call__(
    #     self,
    #     inputs: list[Experiment],
    #     name: str = "",
    #     axis: tp.Literal[0, 1] = 0,
    #     index_name: str | None = None,
    # ) -> Experiment:
    #     ...
    #
    # @tp.overload
    # def __call__(self, inputs: list[Record]) -> Record:
    #     ...
    #
    # @tp.overload
    # def __call__(
    #     self,
    #     inputs: list[Table],
    #     name: str = "",
    #     axis: tp.Literal[0, 1] = 0,
    #     index_name: str | None = None,
    # ) -> Table:
    #     ...

    def __call__(
        self,
        inputs: MappingListTypes,
        name: str = "",
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> MappingT:
        if _is_mapping_list(inputs):
            return super().__call__(inputs, name=name, axis=axis, index_name=index_name)
        else:
            raise TypeError(
                "Type of elements in 'inputs' must be unified either 'Artifact', 'Experiment', 'Record' or 'Table'."
            )

    @tp.overload
    def operate(self, inputs: list[Artifact], name: str = "") -> Artifact:
        ...

    @tp.overload
    def operate(
        self,
        inputs: list[Experiment],
        name: str = "",
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> Experiment:
        ...

    @tp.overload
    def operate(self, inputs: list[Record]) -> Record:
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
        inputs: MappingListTypes,
        name: str = "",
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> MappingTypes:
        if _is_artifact_list(inputs):
            data = {}
            for node in inputs:
                data.update(node.data.copy())
            return type(inputs[0])(data, name)
        elif _is_experiment_list(inputs):
            concat_a: Concat[Artifact] = Concat()
            concat_t: Concat[Table] = Concat()
            artifact = concat_a([n.data[0] for n in inputs])
            table = concat_t([n.data[1] for n in inputs])
            return type(inputs[0])((artifact, table), name)
        elif _is_record_list(inputs):
            data = pd.concat([node.data for node in inputs])
            return type(inputs[0])(data, name)
        elif _is_table_list(inputs):
            data = pd.concat([node.data for node in inputs], axis=axis)
            data.index.name = index_name
            return type(inputs[0])(data, name)
        else:
            raise TypeError(
                "Type of elements in 'inputs' must be unified either 'Artifact', 'Experiment', 'Record' or 'Table'."
            )
