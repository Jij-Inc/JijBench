import os
import inspect
import itertools
import functools
import openjij as oj
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Callable, Union
from jijbench.problems import TSP
from jijbench.experiment import Experiment
from jijbench.evaluation import Evaluator
from jijbench.components import (
    Table,
    Artifact,
    ID,
    ExperimentResultDefaultDir,
)


class BenchmarkSolver:
    _default_solver_names = {
        "SASampler": "openjij_sa_sampler_sample",
        "SQASampler": "openjij_sqa_sampler_sample",
        "JijSASampler": "jijzept_sa_sampler_sample_model",
    }

    def __init__(self, solver):
        self.solver = self._parse_solver(solver)
        self._name = self.solver.__name__
        self._ret_names = (
            ["response", "decoded"] if solver in self._default_solver_names else None
        )

    def __call__(self, **kwargs):
        parameters = inspect.signature(self.solver).parameters
        kwargs = (
            kwargs
            if "kwargs" in parameters
            else {k: v for k, v in kwargs.items() if k in parameters}
        )
        return self.solver(**kwargs)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def ret_names(self):
        return self._ret_names

    @ret_names.setter
    def ret_names(self, names):
        self._ret_names = names

    def to_named_ret(self, ret):
        if isinstance(ret, tuple):
            names = (
                self._ret_names
                if self._ret_names
                else [f"solver_return_values[{i}]" for i in range(len(ret))]
            )
            ret = dict(zip(names, ret))
        else:
            names = self.ret_names if self._ret_names else ["solver_return_values[0]"]
            ret = dict(zip(names, [ret]))
        return ret

    def _parse_solver(self, solver):
        if isinstance(solver, str):
            return getattr(self, self._default_solver_names[solver])
        elif isinstance(solver, Callable):
            return solver
        else:
            return

    @staticmethod
    def on_param_validation(fn):
        @functools.wraps(fn)
        def wrapper(obj, *args, **kwargs):
            (solver,) = args
            if isinstance(solver, (list, tuple)):
                callable_solvers = []
                for s in solver:
                    callable_solvers.append(BenchmarkSolver(s))
                if isinstance(solver, tuple):
                    callable_solvers = tuple(callable_solvers)
            else:
                callable_solvers = [BenchmarkSolver(solver)]
            fn(obj, callable_solvers)

        return wrapper

    @staticmethod
    def openjij_sa_sampler_sample(problem, ph_value, feed_dict=None, **kwargs):
        return BenchmarkSolver._sample(
            oj.SASampler, problem, ph_value, feed_dict, **kwargs
        )

    @staticmethod
    def openjij_sqa_sampler_sample(problem, ph_value, feed_dict=None, **kwargs):
        return BenchmarkSolver._sample(
            oj.SQASampler, problem, ph_value, feed_dict, **kwargs
        )

    @staticmethod
    def _sample(sampler, problem, ph_value, feed_dict, **kwargs):
        if feed_dict is None:
            feed_dict = {const_name: 5.0 for const_name in problem.constraints}
        parameters = inspect.signature(sampler).parameters
        kwargs = {k: w for k, w in kwargs.items() if k in parameters}
        bqm = problem.to_pyqubo(ph_value).compile().to_bqm(feed_dict=feed_dict)
        response = sampler(**kwargs).sample(bqm)
        decoded = problem.decode(response, ph_value=ph_value)
        return response, decoded


class Benchmark:
    def __init__(
        self,
        params,
        *,
        benchmark_id: Union[int, str] = None,
        solver: Any = "SASampler",
        targets=None,
        id_rules: Union[str, Dict[str, str]] = "uuid",
        jijzept_config=None,
        dwave_config=None,
    ):
        self.params = params
        self._id = ID(benchmark_id=benchmark_id)
        self._set_solver(solver)
        if targets is None:
            self.targets = [TSP()]
        else:
            self.targets = targets
        self.id_rules = id_rules
        self._experiments = []
        self._table = Table()
        self._artifact = Artifact()

    @property
    def solver(self):
        return self._solver

    @BenchmarkSolver.on_param_validation
    def _set_solver(self, solver):
        self._solver = solver

    @property
    def experiments(self):
        return self._experiments

    @property
    def table(self):
        return self._table.data

    @property
    def artifact(self):
        return self._artifact.data

    def run(self, show_solver_ret_columns=True):
        keys = list(self.params.keys())
        values = list(self.params.values())

        for target in self.targets:
            for problem, instance in target.parse():
                instance_name, ph_value = instance
                opt_value = ph_value.pop("opt_value", np.nan)
                for solver in self.solver:
                    experiment = Experiment(benchmark_id=self._id.benchmark_id)
                    solver_args = {"problem": problem, "ph_value": ph_value}
                    record = {
                        "problem_name": problem.name,
                        "instance_name": instance_name,
                        "opt_value": opt_value,
                        "solver": solver.name,
                    }
                    for r in itertools.product(*values):
                        with experiment:
                            solver_args |= dict([(k, v) for k, v in zip(keys, r)])
                            ret = solver(**solver_args)
                            ret = solver.to_named_ret(ret)
                            solver_args |= ret
                            record |= dict([(k, v) for k, v in zip(keys, r)])
                            record |= ret
                            experiment.store(record)
                    self._table.data = pd.concat([self._table.data, experiment.table])
                    self._artifact.data |= experiment.artifact
                    self._experiments.append(experiment)

    def _run_by_jijzept():
        pass

    def _run_by_default():
        pass

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
            if k in BenchmarkSolver._default_solver_names:
                new_ret_names[BenchmarkSolver._default_solver_names[k]] = v
            else:
                new_ret_names[k] = v
        for solver in self.solver:
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
