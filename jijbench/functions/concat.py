from __future__ import annotations

import pandas as pd
import pathlib
import typing as tp

from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.elements.id import ID
from jijbench.node.base import DataNode, FunctionNode
from jijbench.typing import MappingT, MappingTypes, MappingListTypes, DataNodeT
from typing_extensions import TypeGuard

if tp.TYPE_CHECKING:
    from jijbench.mappings.mappings import Artifact, Record, Table
    from jijbench.experiment.experiment import Experiment


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


def _is_mapping_list(inputs: MappingListTypes) -> TypeGuard[list[MappingT]]:
    cls_name = inputs[0].__class__.__name__
    return all([node.__class__.__name__ == cls_name for node in inputs])


def _is_datanode_list(inputs: list[DataNodeT]) -> TypeGuard[list[DataNodeT]]:
    sample = inputs[0]
    is_datanode = isinstance(sample, DataNode)
    return all([isinstance(node, sample.__class__) for node in inputs]) & is_datanode


class Concat(FunctionNode[DataNodeT, DataNodeT]):
    """Concat class for concatenating multiple mapping data.
    This class can be apply to `Artifact`, `Experiment`, `Record`, `Table`.
    """

    @tp.overload
    def __call__(self, inputs: list[Artifact], name: tp.Hashable = None) -> Artifact:
        ...

    @tp.overload
    def __call__(
        self,
        inputs: list[Experiment],
        name: str | None = None,
        *,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ) -> Experiment:
        ...

    @tp.overload
    def __call__(self, inputs: list[Record], name: tp.Hashable = None) -> Record:
        ...

    @tp.overload
    def __call__(
        self,
        inputs: list[Table],
        name: tp.Hashable = None,
        *,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> Table:
        ...

    def __call__(
        self,
        inputs: MappingListTypes,
        name: tp.Hashable = None,
        *,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> DataNode:
        """Concatenates the given list of mapping type objects.

        Args:
            inputs (MappingListTypes): A list of artifacts, experiments, records, or tables. The type of elements in 'inputs' must be unified either 'Artifact', 'Experiment', 'Record' or 'Table'.
            name (tp.Hashable, optional): A name for the resulting data. Defaults to None.
            autosave (bool, optional): A flag indicating whether to save the result to disk. Defaults to True.
            savedir (str | pathlib.Path, optional): The directory to save the result in. Defaults to DEFAULT_RESULT_DIR.
            axis (tp.Literal[0, 1], optional): The axis to concatenate the inputs along. Defaults to 0.
            index_name (str | None, optional): The name of the index after concatenation. Defaults to None.

        Raises:
            TypeError: If the type of elements in 'inputs' is not unified either 'Artifact', 'Experiment', 'Record' or 'Table'.

        Returns:
            MappingTypes: The resulting artifact, experiment, record, or table object.
        """
        if _is_mapping_list(inputs):
            return super().__call__(
                inputs,
                name=name,
                autosave=autosave,
                savedir=savedir,
                axis=axis,
                index_name=index_name,
            )
        else:
            raise TypeError(
                "Type of elements in 'inputs' must be unified either 'Artifact', 'Experiment', 'Record' or 'Table'."
            )

    @tp.overload
    def operate(self, inputs: list[Artifact], name: tp.Hashable = None) -> Artifact:
        ...

    @tp.overload
    def operate(
        self,
        inputs: list[Experiment],
        name: str | None = None,
        *,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ) -> Experiment:
        ...

    @tp.overload
    def operate(self, inputs: list[Record]) -> Record:
        ...

    @tp.overload
    def operate(
        self,
        inputs: list[Table],
        name: tp.Hashable = None,
        *,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> Table:
        ...

    def operate(
        self,
        inputs: MappingListTypes,
        name: tp.Hashable = None,
        *,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None,
    ) -> MappingTypes:
        """This method operates the concatenation of the given 'inputs' either 'Artifact', 'Experiment', 'Record' or 'Table'
        objects into a single object of the same type as 'inputs'.

        Args:
            inputs (MappingListTypes): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None. Defaults to None.
            autosave (bool, optional): If True, the resulting object will be saved to disk. Defaults to True.
            savedir (str | pathlib.Path, optional): The directory where the resulting object will be saved. Defaults to DEFAULT_RESULT_DIR.
            axis (tp.Literal[0, 1], optional): The axis along which to concatenate the input 'Table' objects. Defaults to 0.
            index_name (str | None, optional): The name of the resulting object's index. Defaults to None.

        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.

        Returns:
            MappingTypes: The resulting 'Artifact', 'Experiment', 'Record' or 'Table' object.

        """
        if _is_artifact_list(inputs):
            data = {}
            for node in inputs:
                data.update(node.data.copy())
            return type(inputs[0])(data, name)
        elif _is_experiment_list(inputs):
            concat_a: Concat[Artifact] = Concat()
            concat_t: Concat[Table] = Concat()
            inputs_a = [n.data[0] for n in inputs]
            inputs_t = [n.data[1] for n in inputs]
            artifact = inputs_a[0].apply(concat_a, inputs_a[1:])
            table = inputs_t[0].apply(concat_t, inputs_t[1:])

            if name is None:
                name = ID().data

            if not isinstance(name, str):
                raise TypeError("Attirbute name of Experiment must be string.")

            return type(inputs[0])(
                (artifact, table),
                name,
                autosave=autosave,
                savedir=savedir,
            )
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
