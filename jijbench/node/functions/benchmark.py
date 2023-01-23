from __future__ import annotations

import typing as tp
import itertools
import pathlib

from jijbench.const import DEFAULT_RESULT_DIR
from jijbench.node.base import DataNode, FunctionNode
import jijbench.node.data.experiment as _experiment
import jijbench.node.data.id as _id
import jijbench.node.functions.solver as _solver



class Benchmark(FunctionNode[DataNode, _experiment.Experiment]):
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
            name = _id.ID().data
        self._name = name

        self.autosave = autosave
        self.savedir = savedir

    def __call__(self, concurrent=False, extract=True) -> _experiment.Experiment:
        experiment = _experiment.Experiment(
            name=_id.ID().data, autosave=self.autosave, savedir=self.savedir
        )
        for f in self.solver:
            if concurrent:
                experiment = self._run_concurrently(experiment, _solver.Solver(f))
            else:
                experiment = self._run_sequentially(experiment, _solver.Solver(f))
        return experiment

    @property
    def name(self) -> str:
        return self._name

    def _run_concurrently(self, experiment: _experiment.Experiment, solver: _solver.Solver) -> _experiment.Experiment:
        raise NotImplementedError

    def _run_sequentially(
        self, experiment: _experiment.Experiment, solver: _solver.Solver, extract=True
    ) -> _experiment.Experiment:
        # TODO 返り値名を変更できるようにする。
        # solver.rename_return(ret)
        for r in itertools.product(*self.params.values()):
            with experiment:
                solver_args = dict([(k, v) for k, v in zip(self.params.keys(), r)])
                name = _id.ID().data
                record = solver(**solver_args, extract=extract)
                record.name = name
                experiment.append(record)

                # TODO 入力パラメータをtableで保持する
                # params = (dict([(k, v) for k, v in zip(self.params.keys(), r)]))
                # params = RecordFactory().apply(params)
                # params.name = name
                # experiment.append(record)
        return experiment
