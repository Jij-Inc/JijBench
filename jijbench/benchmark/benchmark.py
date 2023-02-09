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
from jijbench.solver.base import Parameter, Solver


class Benchmark(FunctionNode[Experiment, Experiment]):
    """ "A class representing a benchmark.

    This class allows to define a benchmark as a collection of experiments
    over a set of parameters and solvers. The benchmark will be run sequentially
    or concurrently and the results of each experiment will be concatenated and
    returned as a single experiment.
    
    Attributes:
        params (dict[str, Iterable[Any]]): List of lists of parameters for the benchmark.
        solver (Callable | list[Callable]): List of solvers to be used in the benchmark.
        name (str | None, optional): Name of the benchmark.
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
            solver (Callable | list[Callable]): A single solver or a list of solvers to be used in the benchmark.
                The solvers should be callable objects taking in a list of parameters.
            name (str | None, optional): Name of the benchmark. Defaults to None.

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

    def __call__(
        self,
        inputs: list[Experiment] | None = None,
        concurrent: bool = False,
        is_parsed_sampleset: bool = True,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ) -> Experiment:
        """_summary_

        Args:
            inputs (list[Experiment] | None, optional): A list of input experiments to be used by the benchmark. Defaults to None.
            concurrent (bool, optional): Whether to run the experiments concurrently or not. Defaults to False.
            is_parsed_sampleset (bool, optional): Whether the sampleset is parsed or not. Defaults to True.
            autosave (bool, optional): _description_. Whether to automatically save the Experiment object after each run. Defaults to True.
            savedir (str | pathlib.Path, optional): _description_. The directory to save the Experiment object. Defaults to DEFAULT_RESULT_DIR.

        Returns:
            Experiment: _description_
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
        """The name of the benchmark."""
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

    def operate(
        self,
        inputs: list[Experiment],
        concurrent: bool = False,
        is_parsed_sampleset: bool = True,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ) -> Experiment:
        """Performs the operations specified in the benchmark on the input experiments and returns the Experiment object.

        Args:
            inputs (list[Experiment]): A list of input experiments.
            concurrent (bool, optional): Whether to run the operations concurrently or not. Defaults to False.
            is_parsed_sampleset (bool, optional): Whether the sampleset is already parsed or not. Defaults to True.
            autosave (bool, optional): Whether to automatically save the Experiment object after each run. Defaults to True.
            savedir (str | pathlib.Path, optional): The directory to save the Experiment object. Defaults to DEFAULT_RESULT_DIR.

        Returns:
            Experiment: An Experiment object representing the results of the operations performed by the benchmark.
        """
        concat: Concat[Experiment] = Concat()
        name = inputs[0].name
        experiment = concat(inputs, name=name, autosave=autosave, savedir=savedir)
        if concurrent:
            return self._co()
        else:
            return self._seq(experiment, is_parsed_sampleset)

    def _co(self) -> Experiment:
        raise NotImplementedError

    def _seq(
        self,
        experiment: Experiment,
        is_parsed_sampleset: bool,
    ) -> Experiment:
        for f in self.solver:
            for params in self.params:
                with experiment:
                    name = (self.name, experiment.name, ID().data)
                    fdata = [Callable(f.function, str(f.name))]
                    record = f(params, is_parsed_sampleset=is_parsed_sampleset)
                    record = Concat()(
                        [RecordFactory()(params + fdata), record], name=name
                    )
                    experiment.append(record)
            experiment.name = ID().data
        experiment.data[1].index.names = ["benchmark_id", "experiment_id", "run_id"]
        return experiment
