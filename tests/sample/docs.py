def save(
    obj: Artifact | Experiment | Table,
    *,
    savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    mode: tp.Literal["w", "a"] = "w",
    index_col: int | list[int] | None = None,
) -> None:
    """Save the given `Artifact`, `Experiment`, or `Table` object.

    Args:
        obj (Artifact | Experiment | Table): The object to be saved.
        savedir (str or pathlib.Path, optional): The directory where the object will be saved. Defaults to DEFAULT_RESULT_DIR.
        mode (str, optional): The write mode for the file. Must be 'w' or 'a'. Defaults to "w".
        index_col (int or list[int] or None, optional): Index column(s) to set as index while saving the table. Defaults to None.

    Raises:
        ValueError: If the mode is not 'w' or 'a'.
        FileNotFoundError: If the savedir does not exist.
        IOError: If the object is not dillable.
        TypeError: If the object is not an `Artifact`, `Experiment`, or `Table`.
    """
    from jijbench.experiment.experiment import Experiment

    def is_dillable(obj: tp.Any) -> bool:
        try:
            dill.dumps(obj)
            return True
        except Exception:
            return False

    if mode not in ["a", "w"]:
        raise ValueError("Argument mode must be 'a' or 'b'.")

    savedir = savedir if isinstance(savedir, pathlib.Path) else pathlib.Path(savedir)
    if not savedir.exists():
        raise FileNotFoundError(f"Directory {savedir} is not found.")

    if isinstance(obj, Artifact):
        p = savedir / "artifact.dill"
        concat_a: Concat[Artifact] = Concat()
        if not is_dillable(obj):
            raise IOError(f"Cannot save object: {obj}.")

        if mode == "a":
            if p.exists():
                obj = concat_a(
                    [
                        load(
                            savedir,
                            return_type="Artifact",
                        ),
                        obj,
                    ]
                )

        with open(p, "wb") as f:
            dill.dump(obj, f)

    elif isinstance(obj, Experiment):
        savedir_a = savedir / obj.name / "artifact"
        savedir_t = savedir / obj.name / "table"
        savedir_a.mkdir(parents=True, exist_ok=True)
        savedir_t.mkdir(parents=True, exist_ok=True)
        save(
            obj.data[0],
            savedir=savedir_a,
            mode=mode,
        )
        save(
            obj.data[1],
            savedir=savedir_t,
            mode=mode


from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from jijbench.elements.base import Element, Number
from jijbench.functions.math import Min, Max, Mean, Std

@dataclass
class Array(Element[np.ndarray]):
    """A class to store numpy arrays and perform mathematical operations on them.

    Attributes:
        data (np.ndarray): The numpy array to store.
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
        """Validate the input data to ensure it is a numpy array.

        Args:
            data (np.ndarray): The numpy array to validate.

        Returns:
            np.ndarray: The validated numpy array.

        Raises:
            TypeError: If the input data is not a numpy array.
        """
        return cls._validate_dtype(data, (np.ndarray,))


from __future__ import annotations
import typing as tp
import itertools
import pathlib

from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.node.base import FunctionNode
from jijbench.elements.base import Callable
from jijbench.elements.id import ID
from jijbench.experiment.experiment import Experiment
from jijbench.functions.concat import Concat
from jijbench.functions.factory import RecordFactory
from jijbench.solver.solver import Parameter, Solver


class Benchmark(FunctionNode[Experiment, Experiment]):
    """A class representing a benchmark.

    This class allows to define a benchmark as a collection of experiments
    over a set of parameters and solvers. The benchmark will be run sequentially
    or concurrently and the results of each experiment will be concatenated and
    returned as a single experiment.

    Attributes:
        params (List[List[Parameter]]): List of lists of parameters for the benchmark.
        solver (List[Solver]): List of solvers to be used in the benchmark.
        _name (str): Name of the benchmark.

    """

    def __init__(
        self,
        params: dict[str, tp.Iterable[tp.Any]],
        solver: tp.Callable | list[tp.Callable],
        name: str | None = None,
    ) -> None:
        """Initializes the benchmark with the given parameters and solvers.

        Args:
            params (dict[str, Iterable[Any]]): Dictionary of parameters for the benchmark.
                The keys should be the names of the parameters and the values should
                be iterables of the respective parameter values.
            solver (Callable or list[Callable]): A single solver or a list of solvers to be used in the benchmark.
                The solvers should be callable objects taking in a list of parameters.
            name (str, optional): Name of the benchmark. Defaults to None.

        Raises:
            TypeError: If the name is not a string.
        """
        if name is None:
            name = ID().data
        super().__init__(name)

        self.params = [
            [Parameter(v, k) for k, v in zip(params.keys(), r)]
            for r in itertools.product(*params.values())
        ]

        if isinstance(solver, tp.Callable):
            self.solver = [Solver(solver)]
        else:
            self.solver = [Solver(f) for f in solver]

    def __call__(self, inputs: list[Experiment] = None, concurrent: bool = False, is_parsed_sampleset: bool = True, autosave: bool = True, savedir: str = DEFAULT_RESULT_DIR) -> Experiment:
    """Executes the benchmark and returns the Experiment object.

    Args:
        inputs (list[Experiment], optional): A list of input experiments to be used by the benchmark. Defaults to None.
        concurrent (bool, optional): Whether to run the experiments concurrently or not. Defaults to False.
        is_parsed_sampleset (bool, optional): Whether the sampleset is already parsed or not. Defaults to True.
        autosave (bool, optional): Whether to automatically save the Experiment object after each run. Defaults to True.
        savedir (str, optional): The directory to save the Experiment object. Defaults to DEFAULT_RESULT_DIR.

    Returns:
        Experiment: An Experiment object representing the results of the benchmark.
    """
    savedir = (
        savedir if isinstance(savedir, pathlib.Path) else pathlib.Path(savedir)
    )
    savedir /= self.name
    if inputs is None:
        inputs = [Experiment(autosave=autosave, savedir=savedir)]

    return super().__call__(
        inputs,
        concurrent=concurrent,
        is_parsed_sampleset=is_parsed_sampleset,
        autosave=autosave,
        savedir=savedir,
    )

    @property
    def name(self) -> str:
        """str: The name of the benchmark.
        """
        return str(self._name)

    @name.setter
    def name(self, name: str) -> None:
        """Sets the name of the benchmark.

        Args:
            name (str): The name to be set.

        Raises:
            TypeError: If the name is not a string.
        """
        if not isinstance(name, str):
            raise TypeError("Becnhmark name must be string.")
        self._name = name

    def operate(self, inputs: list[Experiment], concurrent: bool = False, is_parsed_sampleset: bool = True, autosave: bool = True, savedir: str = DEFAULT_RESULT_DIR) -> Experiment:
        """Performs the operations specified in the benchmark on the input experiments and returns the Experiment object.

        Args:
            inputs (list[Experiment]): A list of input experiments.
            concurrent (bool, optional): Whether to run the operations concurrently or not. Defaults to False.
            is_parsed_sampleset (bool, optional): Whether the sampleset is already parsed or not. Defaults to True.
            autosave (bool, optional): Whether to automatically save the Experiment object after each run. Defaults to True.
            savedir (str, optional): The directory to save the Experiment object. Defaults to DEFAULT_RESULT_DIR.

        Returns:
            Experiment: An Experiment object representing the results of the operations performed by the benchmark.
        """
        concat: Concat[Experiment] = Concat()
        name = inputs[0].name


from __future__ import annotations
import abc
import copy
import typing as tp

from dataclasses import dataclass, field
from jijbench.typing import T, DataNodeT, DataNodeT2



@dataclass
class DataNode(tp.Generic[T], metaclass=abc.ABCMeta):
    """A base class for all data nodes in a computation graph.
    """
    data: T
    name: tp.Hashable
    operator: FunctionNode | None = field(default=None, repr=False)

    def __setattr__(self, name: str, value: tp.Any) -> None:
        """
        Set the value of an attribute.

        This method is called every time an attribute of the object is set, 
        and it allows for validation of the new value before it is set.

        Args:
            name: The name of the attribute.
            value: The new value of the attribute.
        """
        if name == "data":
            value = self.validate_data(value)
        return super().__setattr__(name, value)

    @property
    def dtype(self) -> type:
        """
        Return the type of the data stored in the node.

        Returns:
            The type of the data stored in the node.

        """
        return type(self.data)

    @classmethod
    @abc.abstractmethod
    def validate_data(cls, data: T) -> T:
        """
        Validate the data stored in the node.

        This method must be implemented by subclasses, and it should raise a TypeError if the
        data is not of a valid type.

        Args:
            data: The data to be validated.

        Returns:
            The data, if it is valid.

        Raises:
            TypeError: If the data is not of a valid type.

        """
        pass

    def apply(self, f: FunctionNode[DataNodeT, DataNodeT2], others: tp.List[DataNodeT] = None, **kwargs: tp.Any) -> DataNodeT2:
        """
        Apply a function `f` on the data stored in the `DataNode` instance and other input `DataNode` instances.

        Args:
            f (FunctionNode[DataNodeT, DataNodeT2]): The function to be applied on the data.
            others (tp.List[DataNodeT], optional): A list of other `DataNode` instances to be used as inputs to the function. Defaults to None.
            **kwargs: Arbitrary keyword arguments to be passed to the function.

        Returns:
            DataNodeT2: The result of applying the function on the data.
        """
        inputs = [tp.cast("DataNodeT", copy.copy(self))] + (others if others else [])
        node = f(inputs, **kwargs)
        node.operator = f
        return node



class FunctionNode(tp.Generic[DataNodeT, DataNodeT2], metaclass=abc.ABCMeta):
    """A base class for all function nodes to operate DataNode objects.

    Args:
        name (tp.Hashable, optional): A name for the function. If not specified, the class name is used.
        inputs (list[DataNodeT]): A list of input `DataNode` objects that the function will operate on.

    Properties:
        name (tp.Hashable): The name of the function.
        inputs (list[DataNodeT]): A list of input `DataNode` objects that the function will operate on.
    """

    def __init__(self, name: tp.Hashable = None) -> None:
        """
        Initialize the function node with a name and an empty list of inputs.
        """
        if name is None:
            name = self.__class__.__name__
        self._name = name
        self.inputs: list[DataNodeT] = []

    def __call__(self, inputs: list[DataNodeT], **kwargs: tp.Any) -> DataNodeT2:
        """Operate on the inputs to produce a new `DataNode` object.

        Args:
            inputs (list[DataNodeT]): A list of input `DataNode` objects.
            kwargs (tp.Any): Keyword arguments for the operation.

        Returns:
            DataNodeT2: A new `DataNode` object that is the result of the operation.
        """
        node = self.operate(inputs, **kwargs)
        self.inputs += inputs
        # node.operator = self
        return node

    @property
    def name(self) -> tp.Hashable:
        """tp.Hashable: The name of the function."""
        return self._name

    @name.setter
    def name(self, name: tp.Hashable) -> None:
        """Set the name of the function.

        Raises:
            TypeError: If the specified name is not hashable.

        Args:
            name (tp.Hashable): The new name for the function.
        """
        if not isinstance(name, tp.Hashable):
            raise TypeError(f"{self.__class__.__name__} name must be hashable.")
        self._name = name

    @abc.abstractmethod
    def operate(self, inputs: list[DataNodeT], **kwargs: tp.Any) -> DataNodeT2:
        """Perform the operation on the inputs.

        This method must be implemented by subclasses.

        Args:
            inputs (list[DataNodeT]): A list of input `DataNode` objects.
            kwargs (tp.Any): Keyword arguments for the operation.

        Returns:
            DataNodeT2: A new `DataNode` object that is the result of the operation.
        """
        pass


from __future__ import annotations

import typing as tp
from dataclasses import dataclass
from jijbench.node.base import DataNode
from jijbench.typing import NumberTypes, T

@dataclass
class Element(DataNode[T]):
    """A generic class representing a data node element.

    Args:
        name (str): The name of the element.
    """
    name: str

@dataclass
class Number(Element[NumberTypes]):
    """A class representing a numerical element.

    Args:
        name (str): The name of the element.
    """

    @classmethod
    def validate_data(cls, data: NumberTypes) -> NumberTypes:
        """Validate the data to be a number.

        Args:
            data (NumberTypes): The data to be validated.

        Returns:
            NumberTypes: The validated data.

        Raises:
            TypeError: If the data is not a number.
        """
        return cls._validate_dtype(data, (int, float))

@dataclass
class String(Element[str]):
    """A class representing a string element.

    Args:
        name (str): The name of the element.
    """

    @classmethod
    def validate_data(cls, data: str) -> str:
        """Validate the data to be a string.

        Args:
            data (str): The data to be validated.

        Returns:
            str: The validated data.

        Raises:
            TypeError: If the data is not a string.
        """
        return cls._validate_dtype(data, (str,))

@dataclass
class Callable(Element[tp.Callable]):
    """A class representing a callable element.

    Args:
        name (str): The name of the element.
    """

    @classmethod
    def validate_data(cls, data: tp.Callable) -> tp.Callable:
        """Validate the data to be a callable.

        Args:
            data (tp.Callable): The data to be validated.

        Returns:
            tp.Callable: The validated data.

        Raises:
            TypeError: If the data is not a callable.
        """
        return cls._validate_dtype(data, (tp.Callable,))



from __future__ import annotations
import datetime
import pandas as pd
from dataclasses import dataclass, field
from jijbench.elements.base import Element
from jijbench.typing import DateTypes

@dataclass
class Date(Element[DateTypes]):
    """Class to store date information.
    
    This class inherits from `Element` class and stores date information in the `data` attribute.
    The data can be stored as a string, datetime.datetime object or pandas.Timestamp object.
    If the data is stored as a string, the class will try to convert it to pandas.Timestamp.

    Args:
        data (DateTypes, optional): The date information to be stored in the node.
        name (str, optional): The name of the node.
    
    Attributes:
        data (DateTypes): The date information stored in the node.
        name (str): The name of the node.
    """

    data: DateTypes = field(default_factory=pd.Timestamp.now)
    name: str = "timestamp"

    def __post_init__(self) -> None:
        """Convert data to pandas.Timestamp if it's a string or datetime.datetime object."""
        if isinstance(self.data, str):
            self.data = pd.Timestamp(self.data)

        if isinstance(self.data, datetime.datetime):
            self.data = pd.Timestamp(self.data)

    @classmethod
    def validate_data(cls, data: DateTypes) -> DateTypes:
        """Validate the type of `data` attribute and make sure it's a string, datetime.datetime object or pandas.Timestamp object.
        
        Raises:
            ValueError: If `data` attribute is a string and not a valid date string.

        Returns:
            DateTypes: The validated date data.
        """
        data = cls._validate_dtype(data, (str, datetime.datetime, pd.Timestamp))
        if isinstance(data, str):
            try:
                pd.Timestamp(data)
            except Exception:
                raise ValueError(f"Date string '{data}' is invalid for data attribute.")
        return data


from __future__ import annotations

import uuid

from dataclasses import dataclass, field
from jijbench.node.base import DataNode


@dataclass
class ID(DataNode[str]):
    """
    Class for a DataNode that holds an ID as data.
    """

    data: str = field(default_factory=lambda: str(uuid.uuid4()))
    """
    The ID data. This is generated as a unique identifier if not specified at the time of instantiation.
    """

    name: str | None = None
    """
    The name for the ID instance.
    """

    @classmethod
    def validate_data(cls, data: str) -> str:
        """
        Validate the data type for the ID.

        Args:
            data: The data to be stored in the instance.

        Returns:
            The validated data.

        Raises:
            TypeError: If the data is not of the expected type.
        """

        return cls._validate_dtype(data, (str,))


from __future__ import annotations

import pandas as pd
import typing as tp
import pathlib
import uuid

from dataclasses import dataclass, field
from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.elements.base import Callable
from jijbench.functions.concat import Concat
from jijbench.functions.factory import ArtifactFactory, TableFactory
from jijbench.io.io import save
from jijbench.mappings.mappings import Artifact, Mapping, Table
from jijbench.solver.solver import Parameter, Return
from jijbench.typing import ExperimentDataType


if tp.TYPE_CHECKING:
    from jijbench.mappings.mappings import Record


@dataclass
class Experiment(Mapping[ExperimentDataType]):
    """
    Experiment class is a subclass of Mapping that specifically holds information for an experiment.

    Attributes:
    data (tuple): A tuple consisting of an Artifact and a Table object that store information about the experiment.
    name (str): A name identifier for the experiment.
    autosave (bool): A flag indicating whether the experiment should automatically save when it is exited.
    savedir (str or pathlib.Path): The directory where the experiment should be saved.
    """
    data: tuple[Artifact, Table] = field(default_factory=lambda: (Artifact(), Table()))
    name: str = field(default_factory=lambda: str(uuid.uuid4()))
    autosave: bool = field(default=True, repr=False)
    savedir: str | pathlib.Path = field(default=DEFAULT_RESULT_DIR, repr=False)

    def __post_init__(self):
    """Initializes the Experiment object."""
        if self.data[0].name is None:
            self.data[0].name = self.name

        if self.data[1].name is None:
            self.data[1].name = self.name

        if isinstance(self.savedir, str):
            self.savedir = pathlib.Path(self.savedir)

    def __enter__(self) -> Experiment:
        """Makes a directory for saving the experiment, if it doesn't exist. Returns the experiment object."""
        savedir = (
            self.savedir
            if isinstance(self.savedir, pathlib.Path)
            else pathlib.Path(self.savedir)
        )
        savedir.mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        """Saves the experiment if autosave is True."""

    @property
    def artifact(self) -> dict:
        """Return the artifact of the experiment as a dictionary.

        Returns:
            dict: The artifact of the experiment.
        """
        return self.view("artifact")

    @property
    def table(self) -> pd.DataFrame:
        """Return the table of the experiment as a pandas dataframe.

        Returns:
            pd.DataFrame: The table of the experiment.
        """
        return self.view("table")

    @property
    def params_table(self) -> pd.DataFrame:
        """Return the parameters table of the experiment as a pandas dataframe.

        Returns:
            pd.DataFrame: The parameters table of the experiment.
        """
        bools = self.data[1].data.applymap(lambda x: isinstance(x, Parameter))
        return self.table[bools].dropna(axis=1)

    @property
    def solver_table(self) -> pd.DataFrame:
        """Return the solver table of the experiment as a pandas dataframe.

        Returns:
            pd.DataFrame: The solver table of the experiment.
        """
        bools = self.data[1].data.applymap(lambda x: isinstance(x, Callable))
        return self.table[bools].dropna(axis=1)

    @property
    def returns_table(self) -> pd.DataFrame:
        """Return the returns table of the experiment as a pandas dataframe.

        Returns:
            pd.DataFrame: The returns table of the experiment.
        """
        bools = self.data[1].data.applymap(lambda x: isinstance(x, Return))
        return self.table[bools].dropna(axis=1)

    @classmethod
    def validate_data(cls, data: ExperimentDataType) -> ExperimentDataType:
        """Validate the data of the experiment.

        Args:
            data (ExperimentDataType): The data to validate.

        Returns:
            ExperimentDataType: The validated data.

        Raises:
            TypeError: If data is not an instance of ExperimentDataType or if the first element of data is not an instance of Artifact or if the second element of data is not an instance of Table.
        """
        artifact, table = data
        if not isinstance(artifact, Artifact):
            raise TypeError(
                f"Type of attribute data is {ExperimentDataType}, and data[0] must be Artifact instead of {type(artifact).__name__}."
            )
        if not isinstance(table, Table):
            raise TypeError(
                f"Type of attribute data is {ExperimentDataType}, and data[1] must be Table instead of {type(artifact).__name__}."
            )
        return data

    def view(self) -> tuple[dict, pd.DataFrame]:
        """Return a tuple of the artifact dictionary and table dataframe.
    
        Returns:
            tuple: A tuple of the artifact dictionary and table dataframe.
        """
        return (self.data[0].view(), self.data[1].view())

    def append(self, record: Record) -> None:
        """Append a new record to the experiment.

        Args:
        record (Record): The record to be appended to the experiment.

        Returns:
        None: This function returns None.
        """
        concat: Concat[Experiment] = Concat()
        data = (ArtifactFactory()([record]), TableFactory()([record]))
        other = type(self)(
            data, self.name, autosave=self.autosave, savedir=self.savedir
        )
        node = self.apply(
            concat,
            [other],
            name=self.name,
            autosave=self.autosave,
            savedir=self.savedir,
        )
        self.__init__(**node.__dict__)

    def save(self):
        """Save the experiment.
        
        Returns:
        None: This function returns None.
        """
        save(self, savedir=self.savedir, mode="a")


@dataclass
class Mapping(DataNode[T]):
    """
    An abstract class for all Mapping classes that implements the methods to be
    followed by all child classes.
    Args:
        None

    Attributes:
        None
    """
    @abc.abstractmethod
    def append(self, record: Record) -> None:
        """
        Append method to be implemented in the child classes.

        Args:
            record: the record to be appended.

        Returns:
            None
        """
        pass

    @abc.abstractmethod
    def view(self) -> T:
        """
        View method to be implemented in the child classes.

        Args:
            None

        Returns:
            A data type of class T.
        """
        pass

        
@dataclass
class Record(Mapping[pd.Series]):
    """Data structure for the Record data.

    Record is a data structure that maps data onto a pandas Series.
    """

    data: pd.Series = field(default_factory=lambda: pd.Series(dtype="object"))
    name: tp.Hashable = None

    @classmethod
    def validate_data(cls, data: pd.Series) -> pd.Series:
        """Validate the input data.

        Validate the input data to ensure that it is a pandas Series and
        all elements of the Series are instances of DataNode.

        Args:
            data (pd.Series): The input data to be validated.

        Returns:
            pd.Series: The validated data.

        Raises:
            TypeError: If the input data is not a pandas Series or
                       if not all elements of the Series are instances of DataNode.
        """
        data = cls._validate_dtype(data, (pd.Series,))
        if data.empty:
            return data
        else:
            if data.apply(lambda x: isinstance(x, DataNode)).all():
                return data
            else:
                raise TypeError(
                    f"All elements of {data.__class__.__name__} must be type DataNode."
                )

    @property
    def index(self) -> pd.Index:
        """Return the index of the Record data.

        Returns:
            pd.Index: The index of the Record data.
        """
        return self.data.index

    @index.setter
    def index(self, index: pd.Index) -> None:
        """Set the index of the Record data.

        Args:
            index (pd.Index): The index to set for the Record data.

        Returns:
            None
        """
        self.data.index = index

    def append(self, record: Record) -> None:
        """Add a new Record to the current Record.

        Args:
            record (Record): The Record to be added.

        Returns:
            None
        """
        concat: Concat[Record] = Concat()
        node = self


@dataclass
class Artifact(Mapping[ArtifactDataType]):
    """A class representing an Artifact.

    Artifacts are metadata containers used to store intermediate results and other
    relevant data produced during the execution of a data processing pipeline. They
    contain metadata about the intermediate data stored and their relationships to other
    Artifacts.

    Attributes:
        data (ArtifactDataType): The data stored in the Artifact.
        name (tp.Hashable, optional): The name of the Artifact. Defaults to None.
    """

    data: ArtifactDataType = field(default_factory=dict)
    name: tp.Hashable = None

    @classmethod
    def validate_data(cls, data: ArtifactDataType) -> ArtifactDataType:
        """Validate the data stored in the Artifact.

        The data stored in the Artifact must be of type `dict`. The values stored in
        the `dict` must be of type `DataNode`.

        Args:
            data (ArtifactDataType): The data to be validated.

        Returns:
            ArtifactDataType: The validated data.

        Raises:
            TypeError: If the data is not of the correct type.
        """
        if data:
            data = cls._validate_dtype(data, (dict,))
            values = []
            for v in data.values():
                if isinstance(v, dict):
                    values += list(v.values())
                else:
                    raise TypeError(
                        f"Type of attibute data is {ArtifactDataType}. Input data is invaid."
                    )
            if all(map(lambda x: isinstance(x, DataNode), values)):
                return data
            else:
                raise TypeError(
                    f"Type of attibute data is {ArtifactDataType}. Input data is invaid."
                )
        else:
            return data

    def keys(self) -> tuple[tp.Hashable, ...]:
        """Return a tuple of the keys of the `data` stored in the Artifact.

        Returns:
            tuple[tp.Hashable, ...]: A tuple of keys.
        """
        return tuple(self.data.keys())

    def values(self) -> tuple[dict[tp.Hashable, DataNode], ...]:
        """Return a tuple of the values of the `data` stored in the Artifact.

        Returns:
            tuple[dict[tp.Hashable, DataNode], ...]: A tuple of values.
        """
        return tuple(self.data.values())

    def items(self) -> tuple[tuple[tp.Hashable, dict[tp.Hashable, DataNode]], ...]:
        """Return a tuple of key-value pairs of the `data` stored in the Artifact.

        Returns:
            tuple[tuple[tp.Hashable, dict[tp.Hashable, DataNode]], ...]: A tuple of
                key-value pairs.
        """
        return tuple(self.data.items())

    def append(self, record: Record) -> None:
        """Append the data in the `record` to the data stored in the Artifact.

        Args:
            record (Record): The data to be appended.
        """
        concat: Concat[Artifact] = Concat()

    def view(self) -> ArtifactDataType:
        """Return the data of each DataNode in the dict as a new dict."""
        return {
            k: {name: node.data for name, node in v.items()}
            for k, v in self.data.items()
        }
        
        
@dataclass
class Table(Mapping[pd.DataFrame]):
    """A class representing a table data structure with pandas DataFrame as its data type.

    This class is one of Mapping. The element in the each cell is a DataNode. Table class extends the basic functionality of a `pandas.DataFrame` with the ability to store and manipulate `DataNode` objects.

    Attributes:
        data (pd.DataFrame): The actual data stored in the table.
        name (tp.Hashable): The name of the Table, which is used as an identifier.
    """
    data: pd.DataFrame = field(default_factory=pd.DataFrame)
    name: tp.Hashable = None

    @classmethod
    def validate_data(cls, data: pd.DataFrame) -> pd.DataFrame:
        """Validate the input data to ensure it is a pandas DataFrame with all elements being of type `DataNode`.
        
        Args:
            data (pd.DataFrame): The input data to validate.
        
        Returns:
            pd.DataFrame: The validated data if it is a pandas DataFrame with all elements being of type `DataNode`.
        
        Raises:
            TypeError: If the input data is not a pandas DataFrame or if it contains elements that are not of type `DataNode`.
        """
        data = cls._validate_dtype(data, (pd.DataFrame,))
        if data.empty:
            return data
        else:
            if data.applymap(lambda x: isinstance(x, DataNode)).values.all():
                return data
            else:
                raise TypeError(
                    f"All elements of {data.__class__.__name__} must be type DataNode."
                )

    @property
    def index(self) -> pd.Index:
        """Get the index of the data stored in the Table object.
        
        Returns:
            pd.Index: The index of the data stored in the Table object.
        """
        return self.data.index

    @index.setter
    def index(self, index: pd.Index) -> None:
        """Set the index of the data stored in the Table object.
        
        Args:
            index (pd.Index): The new index for the data stored in the Table object.
        """
        self.data.index = index

    @property
    def columns(self) -> pd.Index:
        """Get the columns of the data stored in the Table object.
        
        Returns:
            pd.Index: The columns of the data stored in the Table object.
        """
        return self.data.columns

    @columns.setter
    def columns(self, columns: pd.Index) -> None:
        """Set the columns of the data stored in the Table object.
        
        Args:
            columns (pd.Index): The new columns for the data stored in the Table object.
        """
        self.data.columns = columns

    def append(self, record: Record) -> None:
        """Append a new record to the Table.

        Args:
            record (Record): The data to be appended.
        """
        concat: Concat[Table] = Concat()
        other = TableFactory()([record])
        node = self.apply(concat, [other], name=self.name, axis=0)
        self.__init__(**node.__dict__)


    def view(self) -> pd.DataFrame:
        """Return the data of each DataNode in the pandas.DataFrame as a pandas.DataFrame.

        Returns:
            pandas.DataFrame: The view of the data as a pandas DataFrame.
        """
        if self.data.empty:
            return self.data
        else:
            is_tuple_index = all([isinstance(i, tuple) for i in self.data.index])
            if is_tuple_index:
                names = (
                    self.data.index.names if len(self.data.index.names) >= 2 else None
                )
                index = pd.MultiIndex.from_tuples(self.data.index, names=names)
                # TODO 代入しない
                self.data.index = index
            return self.data.applymap(lambda x: x.data)
            
            
class Concat(FunctionNode[MappingT, MappingT]):
    """Concat class for concatenating multiple mapping data.
    This class can be apply to `Artifact`, `Experiment`, `Record`, `Table`.

    Example:
        Concatenating a list of Artifacts:

        >>> concat = Concat()
        >>> artifact1 = Artifact({'a': 1, 'b': 2})
        >>> artifact2 = Artifact({'c': 3, 'd': 4})
        >>> artifact3 = artifact1.apply(concat, [artifact2])

        Concatenating a list of Experiments:

        >>> experiment1 = Experiment(({'a': 1, 'b': 2}, 'table1'))
        >>> experiment2 = Experiment(({'c': 3, 'd': 4}, 'table2'))
        >>> experiment3 = experiment1.apply(concat, [experiment2])

        Concatenating a list of Records:

        >>> record1 = Record([{'a': 1, 'b': 2}])
        >>> record2 = Record([{'c': 3, 'd': 4}])
        >>> record3 = record1.apply(concat, [record2])

        Concatenating a list of Tables:

        >>> table1 = Table([{'a': 1, 'b': 2}])
        >>> table2 = Table([{'c': 3, 'd': 4}])
        >>> table3 = table1.apply(concat, [table2], axis=0)
    """

    @tp.overload
    def __call__(self, inputs: list[Artifact], name: tp.Hashable = None) -> Artifact:
        """Concatenate a list of `Artifact`s and return a single `Artifact`.

        Args:
            inputs (list[Artifact]): A list of `Artifact`s to be concatenated.
            name (tp.Hashable, optional): Name to be assigned to the output `Artifact`.
                Defaults to `None`.

        Returns:
            Artifact: A concatenated `Artifact` obtained by combining the
                input `Artifacts`.

        Raises:
            TypeError: If the type of elements in `inputs` is not `Artifact`.
        """
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
        """Concatenate a list of `Experiments`s and return a single `Experiment`.

        Args:
            inputs (list[Experiment]): A list of `Experiment`s to be concatenated.
            name (str | None, optional): Name to be assigned to the output `Experiment`.
                Defaults to `None`.
            autosave (bool, optional): Flag to indicate whether to save the output `Experiment`.
                Defaults to `True`.
            savedir (str | pathlib.Path, optional): Path where the output `Experiment` will be
        """
    
    def __call__(
        self,
        inputs: MappingListTypes,
        name: tp.Hashable = None,
        *,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
        axis: tp.Literal[0, 1] = 0,
        index_name: str | None = None
    ) -> MappingTypes:
        """Concatenates the given list of mapping type objects.

        Args:
            inputs (MappingListTypes): A list of artifacts, experiments, records, or tables. The type of elements in 'inputs' must be unified either 'Artifact', 'Experiment', 'Record' or 'Table'.
            name (Hashable, optional): A name for the resulting data.
            autosave (bool): A flag indicating whether to save the result to disk. Defaults to True.
            savedir (str | pathlib.Path, optional): The directory to save the result in. Defaults to 'DEFAULT_RESULT_DIR'.
            axis (): The axis to concatenate the inputs along. Defaults to 0.
            index_name: The name of the index after concatenation. Defaults to None.

        Returns:
            The resulting artifact, experiment, record, or table.

        Raises:
            TypeError: If the type of elements in 'inputs' is not unified either 'Artifact', 'Experiment', 'Record' or 'Table'.
        """
        
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
        """
        This method operates the concatenation of the given 'inputs' either 'Artifact', 'Experiment', 'Record' or 'Table' 
        objects into a single object of the same type as 'inputs'.
        
        Args:
            inputs (MappingListTypes): A list of 'Artifact', 'Experiment', 'Record' or 'Table' objects to concatenate.
            name (tp.Hashable, optional): The name of the resulting object. Defaults to None.
            autosave (bool, optional): If True, the resulting object will be saved to disk. Defaults to True.
            savedir (str | pathlib.Path, optional): The directory where the resulting object will be saved. Defaults to
                                                    DEFAULT_RESULT_DIR.
            axis (tp.Literal[0, 1], optional): The axis along which to concatenate the input 'Table' objects. Defaults to 0.
            index_name (str | None, optional): The name of the resulting object's index. Defaults to None.
        
        Returns:
            MappingTypes: The resulting 'Artifact', 'Experiment', 'Record' or 'Table' object resulting from the concatenation
                        of the input 'inputs' objects.
        
        Raises:
            TypeError: If the type of elements in 'inputs' are not unified or if the 'name' attribute is not a string.
        """
        if _is_artifact_list(inputs):
            data = {}
            for node in inputs:
                data.update(node.data.copy())
            return type(inputs[0])(data, name)
        elif _is_experiment_list(


from __future__ import annotations
import abc
import typing as tp

from jijbench.node.base import DataNode, FunctionNode

if tp.TYPE_CHECKING:
    from jijbench.mappings.mappings import Artifact, Record, Table

class Factory(FunctionNode[DataNodeT, DataNodeT2]):
    """An abstract base class for creating a new data node from a list of input nodes.

    Attributes:
        inputs (List[DataNodeT]): List of input data nodes.
        name (str or None): Name of the resulting data node. If not provided, a unique name is assigned.

    """

    @abc.abstractmethod
    def create(self, inputs: list[DataNodeT], name: str | None = None) -> DataNodeT2:
        """Abstract method to create a new data node.

        Subclasses must implement this method.

        Args:
            inputs (List[DataNodeT]): List of input data nodes.
            name (str or None): Name of the resulting data node. If not provided, a unique name is assigned.

        Returns:
            DataNodeT2: The resulting data node.

        """
        pass

    def operate(
        self, inputs: list[DataNodeT], name: str | None = None, **kwargs: tp.Any
    ) -> DataNodeT2:
        """Create a new data node from the given inputs.

        This method calls `create` method to create a new data node from the given inputs.

        Args:
            inputs (List[DataNodeT]): List of input data nodes.
            name (str or None): Name of the resulting data node. If not provided, a unique name is assigned.
            **kwargs: Additional keyword arguments.

        Returns:
            DataNodeT2: The resulting data node.

        """
        return self.create(inputs, name, **kwargs)


class RecordFactory(Factory[DataNodeT, "Record"]):
    """A factory class for creating Record objects.
    
    This class creates Record objects from a list of input DataNode objects. It uses the `create` method to 
    process the input DataNodes, extract their data and convert it into a pandas Series. The resulting Series is
    used to create the Record object. The class also includes a helper method `_to_nodes_from_sampleset` which
    is used to extract the relevant data from jijmodeling SampleSet objects.

    Args:
        Factory: A generic factory class.
        DataNodeT: A type hint for the DataNode objects that will be processed by the factory.
        "Record": A string indicating the class to be created by the factory.

    Attributes:
        None
    """
    def create(
        self,
        inputs: list[DataNodeT],
        name: str = "",
        is_parsed_sampleset: bool = True,
    ) -> Record:
        """Create a Record object from the input DataNode objects.

        This method takes a list of input DataNode objects, processes them and converts them into a pandas 
        Series. The resulting Series is used to create the Record object. If the input data is a jijmodeling 
        SampleSet and `is_parsed_sampleset` is set to True, the relevant data is extracted from the SampleSet 
        using the `_to_nodes_from_sampleset` helper method.

        Args:
            inputs (list[DataNodeT]): A list of DataNode objects to be processed.
            name (str, optional): A name for the Record object. Defaults to "".
            is_parsed_sampleset (bool, optional): Whether to extract data from jijmodeling SampleSet objects. 
                Defaults to True.
        
        Returns:
            Record: A Record object created from the processed input DataNode objects.
        """
        from jijbench.mappings.mappings import Record

        data = {}
        for node in inputs:
            if isinstance(node.data, jm.SampleSet) and is_parsed_sampleset:
                data.update(
                    {n.name: n for n in self._to_nodes_from_sampleset(node.data)}
                )
            else:
                data[node.name] = node
        data = pd.Series(data)
        return Record(data, name)

    def _to_nodes_from_sampleset(self, sampleset: jm.SampleSet) -> list[DataNode]:
        """Extract relevant data from a jijmodeling SampleSet.

        This method extracts relevant data from a jijmodeling SampleSet, such as the number of occurrences,
        energy, objective, constraint violations, number of samples, number of feasible samples, and the
        execution time. The extracted data is returned as a list of DataNode objects.

        Args:
            sampleset (jm.SampleSet): A jijmodeling SampleSet from which to extract data.

        Returns:
            list[DataNode]: A list of DataNode objects containing the extracted data from the SampleSet.
        """
        data = []

        data.append(


class ArtifactFactory(Factory["Record", "Artifact"]):
    """
    Factory class to create an `Artifact` object.

    Attributes:
        None

    Methods:
        create: Creates an `Artifact` object using a list of `Record` inputs.
    """

    def create(self, inputs: list[Record], name: str = "") -> Artifact:
        """
        Creates an `Artifact` object using a list of `Record` inputs.

        Args:
            inputs (list[Record]): A list of `Record` objects to be used to
                create the `Artifact`.
            name (str, optional): Name of the `Artifact` object. Defaults to
                an empty string.

        Returns:
            Artifact: The created `Artifact` object.
        """
        from jijbench.mappings.mappings import Artifact

        data = {
            node.name
            if isinstance(node.name, tp.Hashable)
            else str(node.name): node.data.to_dict()
            for node in inputs
        }
        return Artifact(data, name)


class TableFactory(Factory["Record", "Table"]):
    """
    Factory class to create a `Table` object.

    Attributes:
        None

    Methods:
        create: Creates a `Table` object using a list of `Record` inputs.
    """

    def create(
        self,
        inputs: list[Record],
        name: str = "",
        index_name: str | None = None,
    ) -> Table:
        """
        Creates a `Table` object using a list of `Record` inputs.

        Args:
            inputs (list[Record]): A list of `Record` objects to be used to
                create the `Table`.
            name (str, optional): Name of the `Table` object. Defaults to an
                empty string.
            index_name (str or None, optional): Name of the index in the
                created `Table`. Defaults to None.

        Returns:
            Table: The created `Table` object.
        """
        from jijbench.mappings.mappings import Table

        data = pd.DataFrame({node.name: node.data for node in inputs}).T
        data.index.name = index_name
        return Table(data, name)


from __future__ import annotations
import numpy as np
import typing as tp
from jijbench.elements.base import Number
from jijbench.node.base import FunctionNode

if tp.TYPE_CHECKING:
    from jijbench.elements.array import Array

class Min(FunctionNode["Array", Number]):
    """Calculates the minimum value of an input `Array`.

    The `Min` class is a subclass of `FunctionNode` that calculates the minimum value of an input `Array` 
    and returns the result as a `Number` object.

    Attributes:
        inputs (List[Array]): A list of `Array` objects to operate on.
        name (str): A name for the node.
    """

    def operate(self, inputs: list[Array]) -> Number:
        """Calculates the minimum value of the input `Array`.

        Args:
            inputs (List[Array]): A list of `Array` objects to operate on.

        Returns:
            Number: The result of the calculation as a `Number` object.
        """
        return _operate_array(inputs, np.min)


class Max(FunctionNode["Array", Number]):
    """Calculates the maximum value of an input `Array`.

    The `Max` class is a subclass of `FunctionNode` that calculates the maximum value of an input `Array` 
    and returns the result as a `Number` object.

    Attributes:
        inputs (List[Array]): A list of `Array` objects to operate on.
        name (str): A name for the node.
    """

    def operate(self, inputs: list[Array]) -> Number:
        """Calculates the maximum value of the input `Array`.

        Args:
            inputs (List[Array]): A list of `Array` objects to operate on.

        Returns:
            Number: The result of the calculation as a `Number` object.
        """
        return _operate_array(inputs, np.max)


class Mean(FunctionNode["Array", Number]):
    """Calculates the mean value of an input `Array`.

    The `Mean` class is a subclass of `FunctionNode` that calculates the mean value of an input `Array` 
    and returns the result as a `Number` object.

    Attributes:
        inputs (List[Array]): A list of `Array` objects to operate on.
        name (str): A name for the node.
    """

    def operate(self, inputs: list[Array]) -> Number:
        """Calculates the mean value of the input `Array`.

        Args:
            inputs (List[Array]): A list of `Array` objects to operate on.

        Returns:
            Number: The result of the calculation as a `Number` object.
        """
        return _operate_array(inputs, np.mean)


class Std(FunctionNode["Array", Number]):
    """Calculates the standard deviation of an input `Array`.

    The `Std` class is a subclass of `FunctionNode` that calculates the standard deviation of an input `Array` 
    and returns the result as a `Number` object.

    Attributes:
        inputs
    """


@dataclass
class Parameter(Element[tp.Any]):
    """A parameter for a solver function.
    
    Attributes:
        data (Any): The data in the node.
        name (str): The name of the parameter.
    """

    @classmethod
    def validate_data(cls, data: tp.Any) -> tp.Any:
        """A class method to validate the data before setting it.

        Args:
            data (Any): The data to be validated.

        Returns:
            Any: The validated data.
        """
        return data
        
@dataclass
class Return(Element[tp.Any]):
    """A return value of a solver function.
    Attributes:
        data (Any): The data in the node.
        name (str): The name of the return value.
    """

    @classmethod
    def validate_data(cls, data: tp.Any) -> tp.Any:
        """A class method to validate the data before setting it.

        Args:
            data (Any): The data to be validated.

        Returns:
            Any: The validated data.
        """
        return data

class Solver(FunctionNode[Parameter, Record]):
    """A solver function that takes a list of Parameter and returns a Record.
    Attributes:
        name (str): The name of the solver function.
        function (Callable): The actual function to be executed.
    """

    def __init__(self, function: tp.Callable, name: str | None = None) -> None:
        """The constructor of the `Solver` class.

        Args:
            function (Callable): The actual function to be executed.
            name (str, optional): The name of the solver function. Defaults to None.
        """
        if name is None:
            name = function.__name__
        super().__init__(name)
        self.function = function

    def operate(
        self,
        inputs: list[Parameter],
        is_parsed_sampleset: bool = True,
    ) -> Record:
        """The main operation of the solver function.

        Args:
            inputs (list[Parameter]): The list of input `Parameter` for the solver function.
            is_parsed_sampleset (bool, optional): Whether the sample set is already parsed. Defaults to True.

        Returns:
            Record: The result of the solver function as a `Record`.

        Raises:
            SolverFailedError: If an error occurs inside the solver function.
        """
        parameters = inspect.signature(self.function).parameters
        solver_args = {
            node.name: node.data for node in inputs if node.name in parameters
        }
        try:
            rets = self.function(**solver_args)
            if not isinstance(rets, tuple):
                rets = (rets,)
        except Exception as e
