from __future__ import annotations

import jijmodeling as jm
import numpy as np

from dataclasses import dataclass
from jijbench.solver.base import Parameter
from jijbench.typing import ModelType
from jijmodeling.expression.extract import extract_vars_from_problem


@dataclass
class InstanceData(Parameter[jm.PH_VALUES_INTERFACE]):
    """Instance data for JijZept.

    Attributes:
        data (jijmodeling.PH_VALUES_INTERFACE): A dictionary where keys are the string labels of placeholders in the optimization problem,
            and values are integer, float, list, or numpy.ndarray type data.
        name (str): The name of instance_data.
    """

    @classmethod
    def validate_data(cls, data: jm.PH_VALUES_INTERFACE) -> jm.PH_VALUES_INTERFACE:
        """
        Validate the instance data to ensure it has the correct format.

        Args:
            data (dict): The instance data to validate.

        Raises:
            TypeError: If any key in the data is not of type str.
            TypeError: If any value in the data is not of type int, float, list, or numpy.ndarray.

        Returns:
            dict: The validated instance data.
        """
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
    """User defined model for jijzept.

    Attributes:
        data: A tuple of `Problem` and `PH_VALUES_INTERFACE` which contains the
            problem definition and the instance data.
        name (str): The name of model.
    """

    @classmethod
    def validate_data(cls, data: ModelType) -> ModelType:
        """Validate the instance data in user-defined model.

        Args:
            data: A tuple of `Problem` and `PH_VALUES_INTERFACE` which contains the problem and the instance data.

        Raises:
            KeyError: If any label in the `Problem` is missing from the instance data.

        Returns:
            ModelType: The validated data.
        """
        problem, instance_data = data
        keys = list(instance_data.keys())
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

    @property
    def problem(self) -> jm.Problem:
        """Return the problem in the data of user defined model."""
        return self.data[0]

    @property
    def instance_data(self) -> jm.PH_VALUES_INTERFACE:
        """Return the instance data in the data of user defined model."""
        return self.data[1]
