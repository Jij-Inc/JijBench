import os
import inspect
import itertools
import functools
import openjij as oj
import pandas as pd
from typing import List, Dict, Any, Callable, Union
from jijbench import problems
from jijbench.experiment import Experiment
from jijbench.objects.id import ID
from jijbench.objects.dir import ExperimentResultDefaultDir
from jijbench.objects.table import Table
from jijbench.objects.artifact import Artifact


class BenchmarkSolver:
    _default_solver_names = {
        "SASampler": "openjij_sa_sampler_sample",
        "SQASampler": "openjij_sqa_sampler_sample",
        "CSQASampler": "openjij_csqa_sampler_sample",
    }

    def __init__(self, solver):
        self.solver = self._parse_solver(solver)
        self._name = self.solver.__name__

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

    # エラーが出る
    @staticmethod
    def openjij_csqa_sampler_sample(problem, ph_value, feed_dict=None, **kwargs):
        return BenchmarkSolver._sample(
            oj.CSQASampler, problem, ph_value, feed_dict, **kwargs
        )

    @staticmethod
    def _sample(sampler, problem, ph_value, feed_dict, **kwargs):
        if feed_dict is None:
            feed_dict = {const_name: 5.0 for const_name in problem.constraints}
        parameters = inspect.signature(sampler).parameters
        kwargs = {k: w for k, w in kwargs.items() if k in parameters}
        bqm = problem.to_pyqubo(ph_value).compile().to_bqm(feed_dict=feed_dict)
        return sampler(**kwargs).sample(bqm)


class Benchmark:
    def __init__(
        self,
        params,
        *,
        benchmark_id: Union[int, str] = None,
        solver: Any = "SASampler",
        targets=None,
        id_rules: Union[str, Dict[str, str]] = "uuid",
    ):
        self.params = params
        self._id = ID(benchmark_id=benchmark_id)
        self._set_solver(solver)
        if targets is None:
            tsp_instance = problems.tsp.tsp_instance()
            name = tsp_instance.small_list()[0]
            data = tsp_instance.get_instance("small", name)
            self.targets = {problems.tsp.travelling_salesman(): (name, data)}
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

    def run(self, solver_ret_names=None):
        keys = list(self.params.keys())
        values = list(self.params.values())

        for solver in self.solver:
            for problem, instance in self.targets.items():
                experiment = Experiment(benchmark_id=self._id.benchmark_id)
                instance_name, ph_value = instance
                kwargs = {"problem": problem, "ph_value": ph_value}
                record = {
                    "problem_name": problem.name,
                    "instance_name": instance_name,
                    "solver": solver.name,
                }
                for r in itertools.product(*values):
                    with experiment:
                        kwargs |= dict([(k, v) for k, v in zip(keys, r)])
                        ret = solver(**kwargs)
                        ret = self._name_solver_ret(ret, solver_ret_names)
                        record |= dict([(k, v) for k, v in zip(keys, r)])
                        record |= ret
                        experiment.store(record)
                        kwargs |= self._parse_ret(ret, kwargs)
                self._table.data = pd.concat([self._table.data, experiment.table])
                self._artifact.data |= experiment.artifact
                self._experiments.append(experiment)

    def compare(self, key, values=None):
        return self._table.data.pivot(columns=key, values=values)

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

    @staticmethod
    def _name_solver_ret(ret, names):
        if isinstance(ret, dict):
            return ret
        elif isinstance(ret, (list, tuple)):
            names = (
                names
                if names
                else [f"solver_return_values[{i}]" for i in range(len(ret))]
            )
            ret = dict(zip(names, ret))
        else:
            ret = {"solver_return_values[0]": ret}
        return ret

    @staticmethod
    def _parse_ret(ret, solver_args):
        return {k: v for k, v in ret.items() if k in solver_args}


class BenchmarkSolver:
    _default_solver_names = {
        "SASampler": "openjij_sa_sampler_sample",
        "SQASampler": "openjij_sqa_sampler_sample",
        "CSQASampler": "openjij_csqa_sampler_sample",
    }

    def __init__(self, solver):
        self.solver = self._parse_solver(solver)
        self._name = self.solver.__name__

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

    # エラーが出る
    @staticmethod
    def openjij_csqa_sampler_sample(problem, ph_value, feed_dict=None, **kwargs):
        return BenchmarkSolver._sample(
            oj.CSQASampler, problem, ph_value, feed_dict, **kwargs
        )

    @staticmethod
    def _sample(sampler, problem, ph_value, feed_dict, **kwargs):
        if feed_dict is None:
            feed_dict = {const_name: 5.0 for const_name in problem.constraints}
        parameters = inspect.signature(sampler).parameters
        kwargs = {k: w for k, w in kwargs.items() if k in parameters}
        bqm = problem.to_pyqubo(ph_value).compile().to_bqm(feed_dict=feed_dict)
        return sampler(**kwargs).sample(bqm)
