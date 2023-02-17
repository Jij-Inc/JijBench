from __future__ import annotations

import numpy as np


from dataclasses import dataclass
from jijbench.elements.base import Number
from jijbench.functions.math import Min, Max, Mean, Std
from jijbench.node.base import DataNode


@dataclass
class Array(DataNode[np.ndarray]):
    """A class representing numpy arrays.
    
    Attributes:
        data (numpy.ndarray): The numpy array.
        name (str): The name of the node.
    """
    def min(self) -> Number:
        """Get the minimum value of the numpy array.

        Returns:
            Number: The minimum value of the numpy array.
        """
        return self.apply(Min())

    def max(self) -> Number:
        """Get the maximum value of the numpy array.

        Returns:
            Number: The maximum value of the numpy array.
        """
        return self.apply(Max())

    def mean(self) -> Number:
        """Get the mean value of the numpy array.

        Returns:
            Number: The mean value of the numpy array.
        """
        return self.apply(Mean())

    def std(self) -> Number:
        """Get the standard deviation of the numpy array.

        Returns:
            Number: The standard deviation of the numpy array.
        """
        return self.apply(Std())

    @classmethod
    def validate_data(cls, data: np.ndarray) -> np.ndarray:
        """Validate the data to ensure it is a numpy array.

        Args:
            data (numpyp.ndarray): The numpy array to validate.
        
        Raises:
            TypeError: If the input data is not a numpy array.

        Returns:
            numpy.ndarray: The validated numpy array.
        """
        return cls._validate_dtype(data, (np.ndarray,))
