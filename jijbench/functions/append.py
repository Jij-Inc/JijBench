from __future__ import annotations

import typing as tp

from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, TableFactory
from jijbench.typing import MappingTypes
from typing_extensions import TypeGuard

if tp.TYPE_CHECKING:
    from jijbench.data.mapping import Artifact, Record, Table
    from jijbench.experiment.experiment import Experiment


def _is_artifact(
    node: MappingTypes,
) -> TypeGuard[Artifact]:
    return node.__class__.__name__ == "Artifact"


def _is_experiment(
    node: MappingTypes,
) -> TypeGuard[Experiment]:
    return node.__class__.__name__ == "Experiment"


def _is_record(
    node: MappingTypes,
) -> TypeGuard[Record]:
    return node.__class__.__name__ == "Record"


def _is_table(
    node: MappingTypes,
) -> TypeGuard[Table]:
    return node.__class__.__name__ == "Table"


@tp.overload
def append(node: Artifact, record: Record) -> None:
    ...


@tp.overload
def append(
    node: Experiment,
    record: Record,
    axis: tp.Literal[0, 1] = 0,
    index_name: tp.Any | None = None,
):
    ...


@tp.overload
def append(node: Record, record: Record) -> None:
    ...


@tp.overload
def append(
    node: Table,
    record: Record,
    axis: tp.Literal[0, 1] = 0,
    index_name: tp.Any | None = None,
) -> None:
    ...


def append(
    node: MappingTypes,
    record: Record,
    axis: tp.Literal[0, 1] = 0,
    index_name: tp.Any | None = None,
) -> None:
    concat = Concat()
    if _is_artifact(node):
        others = [ArtifactFactory()([record])]
        node.data = node.apply(concat, others).data
    elif _is_experiment(node):
        node.data[0].append(record)
        node.data[1].append(record, axis, index_name)
    elif _is_record(node):
        node.data = node.apply(concat, [record]).data
    elif _is_table(node):
        others = [TableFactory()([record])]
        node.data = node.apply(concat, others).data
    else:
        raise TypeError(f"{node.__class__.__name__} does not support 'append'.")
    node.operator = concat
