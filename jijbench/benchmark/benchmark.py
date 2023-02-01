from __future__ import annotations

import typing as tp
import itertools
import pandas as pd
import pathlib


from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.node.base import FunctionNode
from jijbench.data.elements.id import ID
from jijbench.data.elements.values import Callable, Parameter
from jijbench.experiment.experiment import Experiment
from jijbench.functions.concat import Concat
from jijbench.functions.factory import RecordFactory
from jijbench.functions.solver import Solver


class Benchmark(FunctionNode[Experiment, Experiment]):
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
        inputs: list[Experiment] | None = None,
        concurrent: bool = False,
        is_parsed_sampleset: bool = True,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ) -> Experiment:
        if inputs is None:
            inputs = [Experiment(name=ID().data, autosave=autosave, savedir=savedir)]

        return super().__call__(
            inputs,
            concurrent=concurrent,
            is_parsed_sampleset=is_parsed_sampleset,
            autosave=autosave,
            savedir=savedir,
        )

    def operate(
        self,
        inputs: list[Experiment],
        concurrent: bool = False,
        is_parsed_sampleset: bool = True,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ) -> Experiment:
        concat: Concat[Experiment] = Concat()
        experiment = concat(inputs, name=self.name, autosave=autosave, savedir=savedir)
        if concurrent:
            return self._co()
        else:
            return self._seq(experiment, is_parsed_sampleset)

    @property
    def name(self) -> str:
        return self._name

    def _co(self) -> Experiment:
        raise NotImplementedError

    def _seq(self, experiment: Experiment, is_parsed_sampleset: bool) -> Experiment:
        # TODO 返り値名を変更できるようにする。
        # solver.rename_return(ret)
        # name = ID().data
        # record = solver(inputs, is_parsed_sampleset=is_parsed_sampleset)
        # record.name = name
        # TODO 入力パラメータをtableで保持する
        # params = (dict([(k, v) for k, v in zip(self.params.keys(), r)]))
        # params = RecordFactory().apply(params)
        # params.name = name
        # experiment.append(record)
        # return Experiment()
        names = []
        for f in self.solver:
            for params in self.params:
                with experiment:
                    name = ID().data
                    fdata = [Callable(f.function, f.name)]
                    record = f(params, is_parsed_sampleset=is_parsed_sampleset)
                    record = Concat()(
                        [RecordFactory()(params + fdata), record], name=name
                    )
                    experiment.append(record)
                    names.append((experiment.name, name))
        index = pd.MultiIndex.from_tuples(names, names=("experiment_id", "run_id"))
        experiment.data[1].data.index = index
        return experiment
