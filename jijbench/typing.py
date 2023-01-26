from __future__ import annotations

import typing as tp

from typing_extensions import TypeAlias

if tp.TYPE_CHECKING:
    from jijbench.data.mapping import Artifact, Record, Table
    from jijbench.experiment.experiment import Experiment
    from jijbench.node.base import DataNode

DataNodeIT = tp.TypeVar("DataNodeIT", bound="DataNode")
DataNodeOT = tp.TypeVar("DataNodeOT", bound="DataNode")
DataNodeIT_co = tp.TypeVar("DataNodeIT_co", bound="DataNode", covariant=True)
DataNodeOT_co = tp.TypeVar("DataNodeOT_co", bound="DataNode", covariant=True)

MappingTypes: TypeAlias = tp.Union["Artifact", "Experiment", "Record", "Table"]
MappingListTypes: TypeAlias = tp.Union[
    list["Artifact"], list["Experiment"], list["Record"], list["Table"]
]
