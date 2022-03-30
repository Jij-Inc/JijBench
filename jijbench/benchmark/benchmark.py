import os
import inspect
import itertools
import jijzept as jz
import numpy as np
import pandas as pd
from typing import List, Dict, Union
from jijmodeling.exceptions import DataError
from jijbench.experiment import Experiment
from jijbench.evaluation import Evaluator
from jijbench.components import (
    Table,
    Artifact,
    ID,
    ExperimentResultDefaultDir,
)
from jijbench.solver import DefaultSolver
from jijbench.benchmark import validation


class Benchmark:
    def __init__(
        self,
        params,
        solvers,
        targets,
        *,
        benchmark_id: Union[int, str] = None,
        id_rules: Union[str, Dict[str, str]] = "uuid",
        jijzept_config=None,
        dwave_config=None,
    ):
        self.params = params
        self.id_rules = id_rules

        self._set_solver(solvers)
        self._set_targets(targets)
        self._id = ID(benchmark_id=benchmark_id)
        self._experiments = []
        self._table = Table()
        self._artifact = Artifact()

        DefaultSolver.jijzept_config = jijzept_config
        DefaultSolver.dwave_config = dwave_config

    @property
    def solvers(self):
        return self._solver

    @validation.on_solver
    def _set_solver(self, solvers):
        self._solver = solvers

    @property
    def targets(self):
        return self._targets

    @validation.on_target
    def _set_targets(self, targets):
        self._targets = targets

    @property
    def experiments(self):
        return self._experiments

    @property
    def table(self):
        return self._table.data

    @property
    def artifact(self):
        return self._artifact.data

    def run(self, show_solver_ret_columns=True, sync=True):
        for target in self.targets:
            for problem, instance in target.parse():
                for solver in self.solvers:
                    if sync:
                        self._run_by_sync(solver, problem, instance)
                    else:
                        if "openjij" in solver.name:
                            self._run_by_sync(solver, problem, instance)
                        else:
                            self._run_by_async(solver, problem, instance)

    def _run_by_async(self, solver, problem, instance, **kwargs):
        _, ph_value = instance
        idx = [
            s in inspect.getsource(solver.function)
            for s in DefaultSolver.jijzept_sampler_names
        ]
        solution_ids = []
        if any(idx):
            for i_name, b in enumerate(idx):
                if b:
                    name = DefaultSolver.jijzept_sampler_names[i_name]
                    sampler = getattr(jz.sampler, name)(
                        config=DefaultSolver.jijzept_config
                    )
                    parameters = inspect.signature(sampler.sample_model).parameters
                    for i, r in enumerate(itertools.product(*self.params.values())):
                        solver_args = dict(
                            [(k, v) for k, v in zip(self.params.keys(), r)]
                        )
                        solver_args = {
                            k: w for k, w in solver_args.items() if k in parameters
                        }
                        solution_ids.append(
                            (
                                i,
                                sampler.sample_model(
                                    problem, ph_value, sync=False, **solver_args
                                ).solution_id,
                            )
                        )
                    experiment = Experiment(benchmark_id=self._id.benchmark_id)
                    solver_args, record = self._setup_experiment(
                        solver, problem, instance, False
                    )
                    for i, solution_id in solution_ids:
                        solver_args |= {"solution_id": solution_id}
                        while True:
                            try:
                                ret = solver(**solver_args)
                                if "APIStatus.SUCCESS" in str(ret):
                                    with experiment:
                                        ret = solver.to_named_ret(ret)
                                        record |= dict(
                                            [(k, v) for k, v in zip(self.params.keys(), r)]
                                        )
                                        record |= {"i": i, "solution_id": solution_id} | ret
                                        experiment.store(record)
                                    break
                            except DataError:
                                pass
                    experiment.table.sort_values("i", inplace=True)
                    experiment.table.drop("i", axis=1, inplace=True)

            self._table.data = pd.concat([self._table.data, experiment.table])
            self._artifact.data |= experiment.artifact
            self._experiments.append(experiment)

    def _run_by_sync(self, solver, problem, instance):
        experiment = Experiment(benchmark_id=self._id.benchmark_id)
        solver_args, record = self._setup_experiment(solver, problem, instance, True)
        for r in itertools.product(*self.params.values()):
            with experiment:
                solver_args |= dict([(k, v) for k, v in zip(self.params.keys(), r)])
                ret = solver(**solver_args)
                ret = solver.to_named_ret(ret)
                solver_args |= ret
                record |= dict([(k, v) for k, v in zip(self.params.keys(), r)])
                record |= ret
                experiment.store(record)
        self._table.data = pd.concat([self._table.data, experiment.table])
        self._artifact.data |= experiment.artifact
        self._experiments.append(experiment)

    @staticmethod
    def _setup_experiment(solver, problem, instance, sync):
        instance_name, ph_value = instance
        opt_value = ph_value.pop("opt_value", np.nan)

        solver_args = {"problem": problem, "ph_value": ph_value, "sync": sync}
        record = {
            "problem_name": problem.name,
            "instance_name": instance_name,
            "opt_value": opt_value,
            "solver": solver.name,
        }
        return solver_args, record

    def compare(self, key, values=None):
        return self._table.data.pivot(columns=key, values=values)

    def evaluate(self, opt_value=None, pr=0.99, expand=True):
        table = Table()
        metrics = pd.DataFrame()
        for experiment in self._experiments:
            evaluator = Evaluator(experiment)
            opt_value = experiment.table["opt_value"][0]
            metrics = pd.concat(
                [
                    metrics,
                    evaluator.calc_typical_metrics(
                        opt_value=opt_value, pr=pr, expand=expand
                    ),
                ]
            )
            table.data = pd.concat([table.data, experiment.table])
        self._table = table
        return metrics

    def name_solver_ret(self, ret_names):
        new_ret_names = {}
        for k, v in ret_names.items():
            if k in dir(DefaultSolver):
                new_ret_names[getattr(DefaultSolver(), k).__name__] = v
            else:
                new_ret_names[k] = v
        for solver in self.solvers:
            if solver.ret_names is None:
                solver.ret_names = (
                    new_ret_names[solver.name] if solver.name in new_ret_names else None
                )

    @classmethod
    def load(
        cls,
        *,
        benchmark_id: Union[int, str],
        experiment_id: Union[int, str, List[Union[int, str]]] = None,
        autosave: bool = True,
        save_dir: str = ExperimentResultDefaultDir,
    ):
        experiments = []
        table = Table()
        artifact = Artifact()
        experiment_ids = (
            experiment_id
            if experiment_id
            else os.listdir(f"{save_dir}/benchmark_{benchmark_id}")
        )
        for experiment_id in experiment_ids:
            experiment = Experiment.load(
                benchmark_id=benchmark_id,
                experiment_id=experiment_id,
                autosave=autosave,
                save_dir=save_dir,
            )
            experiments.append(experiment)
            table.data = pd.concat([table.data, experiment.table])
            artifact.data |= experiment.artifact

        bench = cls([], benchmark_id=benchmark_id)
        bench._experiments = experiments
        bench._table = table
        bench._artifacat = artifact
        return bench
