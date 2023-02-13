from __future__ import annotations

import jijmodeling as jm
import numpy as np

from dataclasses import dataclass
from jijbench.solver.base import Parameter
from jijbench.typing import ModelType
from jijmodeling.expression.extract import extract_vars_from_problem


@dataclass
class InstanceData(Parameter[jm.PH_VALUES_INTERFACE]):
    @classmethod
    def validate_data(cls, data: jm.PH_VALUES_INTERFACE) -> jm.PH_VALUES_INTERFACE:
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
            invalid_values = [
                v for b, v in zip(is_instance_data_values, data.values()) if not b
            ]
            raise TypeError(
                f"The following value(s) {invalid_values} of instance data is invalid. The type of value must be int, float, list, or numpy.ndarray."
            )
        return data


@dataclass
class UserDefinedModel(Parameter[ModelType]):
    @classmethod
    def validate_data(cls, data: ModelType) -> ModelType:
        problem, instance_data = data
        keys = list(instance_data.data.keys())
        ph_labels = [
            v.label
            for v in extract_vars_from_problem(problem)
            if isinstance(v, jm.Placeholder)
        ]

        is_in_labels = list(map(lambda x: x in keys, ph_labels))
        if not all(is_in_labels):
            missing_labels = [p for b, p in zip(is_in_labels, ph_labels) if not b]
            raise KeyError(
                f"Instance data needs label(s) {missing_labels}, but are not included."
            )
        return data
