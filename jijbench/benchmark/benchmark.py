from __future__ import annotations

import typing as tp
import itertools
import pathlib


from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.node.base import DataNode, FunctionNode
from jijbench.experiment.experiment import Experiment
from jijbench.data.elements.id import ID
from jijbench.functions.solver import Solver


class Benchmark(FunctionNode[Experiment]):
    def __init__(
        self,
        params: dict[str, tp.Iterable[tp.Any]],
        solver: tp.Callable | list[tp.Callable],
        name: str | None = None,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ) -> None:
        super().__init__()
        self.params = params
        if isinstance(solver, tp.Callable):
            self.solver = [solver]
        else:
            self.solver = solver

        if name is None:
            name = ID().data
        self._name = name

        self.autosave = autosave
        self.savedir = savedir

    def __call__(
        self, concurrent: bool = False, is_parsed_sampleset: bool = True
    ) -> Experiment:
        experiment = Experiment(
            name=ID().data, autosave=self.autosave, savedir=self.savedir
        )
        for f in self.solver:
            if concurrent:
                experiment = self._run_co(experiment, Solver(f))
            else:
                experiment = self._run_seq(
                    experiment, Solver(f), is_parsed_sampleset=is_parsed_sampleset
                )
        return experiment

    @property
    def name(self) -> str:
        return self._name

    def _run_co(self, experiment: Experiment, solver: Solver) -> Experiment:
        raise NotImplementedError

    def _run_seq(
        self, experiment: Experiment, solver: Solver, is_parsed_sampleset=True
    ) -> Experiment:
        # TODO 返り値名を変更できるようにする。
        # solver.rename_return(ret)
        for r in itertools.product(*self.params.values()):
            with experiment:
                solver_args = dict([(k, v) for k, v in zip(self.params.keys(), r)])
                name = ID().data
                record = solver(**solver_args, is_parsed_sampleset=is_parsed_sampleset)
                record.name = name
                experiment.append(record)

                # TODO 入力パラメータをtableで保持する
                # params = (dict([(k, v) for k, v in zip(self.params.keys(), r)]))
                # params = RecordFactory().apply(params)
                # params.name = name
                # experiment.append(record)
        return experiment
