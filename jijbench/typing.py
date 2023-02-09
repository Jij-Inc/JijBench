from __future__ import annotations

import datetime
import jijmodeling as jm
import numpy as np
import pandas as pd
import typing as tp

from typing_extensions import TypeAlias

if tp.TYPE_CHECKING:
    from jijbench.mappings.mappings import Artifact, Record, Table
    from jijbench.experiment.experiment import Experiment
    from jijbench.node.base import DataNode

T = tp.TypeVar("T")

DataNodeT = tp.TypeVar("DataNodeT", bound="DataNode")
DataNodeT2 = tp.TypeVar("DataNodeT2", bound="DataNode")
DataNodeT_co = tp.TypeVar("DataNodeT_co", bound="DataNode", covariant=True)
DataNodeT2_co = tp.TypeVar("DataNodeT2_co", bound="DataNode", covariant=True)

ArtifactDataType: TypeAlias = tp.Dict[tp.Hashable, tp.Dict[tp.Hashable, "DataNode"]]
ExperimentDataType: TypeAlias = tp.Tuple["Artifact", "Table"]
DateTypes: TypeAlias = tp.Union[str, datetime.datetime, pd.Timestamp]
NumberTypes: TypeAlias = tp.Union[int, float]

InstanceDataType: TypeAlias = tp.Dict[
    str, tp.Union[int, float, tp.List, np.ndarray[tp.Tuple, np.dtype]]
]
ModelType: TypeAlias = tp.Tuple[jm.Problem, InstanceDataType]

MappingT = tp.TypeVar("MappingT", "Artifact", "Experiment", "Record", "Table")
MappingTypes: TypeAlias = tp.Union["Artifact", "Experiment", "Record", "Table"]
MappingListTypes: TypeAlias = tp.Union[
    tp.List["Artifact"], tp.List["Experiment"], tp.List["Record"], tp.List["Table"]
]
