from __future__ import annotations

import typing as tp
import itertools
import pathlib


from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.node.base import FunctionNode
from jijbench.data.elements.id import ID
from jijbench.data.elements.values import Parameter
from jijbench.experiment.experiment import Experiment
from jijbench.functions.solver import Solver


class Benchmark(FunctionNode[Parameter, Experiment]):
    def __init__(
        self,
        params: dict[str, tp.Iterable[tp.Any]],
        solver: tp.Callable | list[tp.Callable],
        name: str | None = None,
    ) -> None:
        super().__init__()

        self.params = [
            [Parameter(v, k) for k, v in zip(params.keys(), r)]
            for r in itertools.product(*params.values())
        ]

        if isinstance(solver, tp.Callable):
            self.solver = [Solver(solver)]
        else:
            self.solver = [Solver(f) for f in solver]

        if name is None:
            name = ID().data
        self._name = name

    # TODO インターフェースを統一
    def __call__(
        self,
        concurrent: bool = False,
        is_parsed_sampleset: bool = True,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ) -> Experiment:
        return self.operate(concurrent, is_parsed_sampleset, autosave, savedir)

    def operate(
        self,
        concurrent: bool = False,
        is_parsed_sampleset: bool = True,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ) -> Experiment:
        if concurrent:
            return self._run_co()
        else:
            experiment = Experiment(name=ID().data, autosave=autosave, savedir=savedir)
            for f in self.solver:
                for inputs in self.params:
                    with experiment:
                        record = f(inputs, is_parsed_sampleset=is_parsed_sampleset)
                        record.name = ID().data
                        experiment.append(record)
            return experiment

    @property
    def name(self) -> str:
        return self._name

    def _run_co(self) -> Experiment:
        raise NotImplementedError

    def _run_seq(
        self, inputs: list[Parameter], solver: Solver, is_parsed_sampleset=True
    ) -> Experiment:
        # TODO 返り値名を変更できるようにする。
        # solver.rename_return(ret)
        name = ID().data
        record = solver(inputs, is_parsed_sampleset=is_parsed_sampleset)
        record.name = name
        # TODO 入力パラメータをtableで保持する
        # params = (dict([(k, v) for k, v in zip(self.params.keys(), r)]))
        # params = RecordFactory().apply(params)
        # params.name = name
        # experiment.append(record)
        return Experiment()
