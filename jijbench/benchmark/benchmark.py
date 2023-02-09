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
    """Executes the benchmark.

    Args:
        params (dict[str, tp.Iterable[tp.Any]]): Parameters to be swept in the benchmark. key is parameter name. list is list of value of a parameter.
        solver (tp.Callable | list[tp.Callable]): Callable solver or list of the solves.
        name (str | None, optional): Becnhmark name.
    """

    def __init__(
        self,
        params: dict[str, tp.Iterable[tp.Any]],
        solver: tp.Callable | list[tp.Callable],
        name: str | None = None,
    ) -> None:
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
            inputs (list[Experiment] | None, optional): _description_. Defaults to None.
            concurrent (bool, optional): _description_. Defaults to False.
            is_parsed_sampleset (bool, optional): _description_. Defaults to True.
            autosave (bool, optional): _description_. Defaults to True.
            savedir (str | pathlib.Path, optional): _description_. Defaults to DEFAULT_RESULT_DIR.

        Returns:
            Experiment: _description_
        """
        savedir = savedir if isinstance(savedir, pathlib.Path) else pathlib.Path(savedir)
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
        return str(self._name)

    @name.setter
    def name(self, name: str) -> None:
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
        """_summary_

        Args:
            inputs (list[Experiment]): _description_
            concurrent (bool, optional): _description_. Defaults to False.
            is_parsed_sampleset (bool, optional): _description_. Defaults to True.
            autosave (bool, optional): _description_. Defaults to True.
            savedir (str | pathlib.Path, optional): _description_. Defaults to DEFAULT_RESULT_DIR.

        Returns:
            Experiment: _description_
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
                    record = Concat()([RecordFactory()(params + fdata), record], name=name)
                    experiment.append(record)
            experiment.name = ID().data
        experiment.data[1].index.names = ["benchmark_id", "experiment_id", "run_id"]
        return experiment
