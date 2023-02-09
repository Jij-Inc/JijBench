from __future__ import annotations

import numpy as np

from dataclasses import dataclass
from jijbench.node.base import DataNode
from jijbench.typing import InstanceDataType, ModelType


@dataclass
class InstanceData(DataNode[InstanceDataType]):
    @classmethod
    def validate_data(cls, data: InstanceDataType) -> InstanceDataType:
        data = cls._validate_dtype(data, (dict,))

        is_instance_data_keys = [isinstance(k, str) for k in data]
        if not all(is_instance_data_keys):
            invalid_keys = [k for b, k in zip(is_instance_data_keys, data) if not b]
            raise TypeError(
                f"The following key(s) {invalid_keys} of instance data is invalid. The type of key must be str."
            )

        is_instance_data_values = [
            isinstance(v, (int, float, list, np.ndarray)) for v in data.values()
        ]
        if not all(is_instance_data_values):
            invalid_values = [v for b, v in zip(is_instance_data_values, data.values()) if not b]
            raise TypeError(
                f"The following value(s) {invalid_values} of instance data is invalid. The type of value must be int, float, list, or numpy.ndarray."
            )
        return data


@dataclass
class UserDefinedModel(DataNode[ModelType]):
    @classmethod
    def validate_data(cls, data: ModelType) -> ModelType:
        problem, instance_data = data
        
