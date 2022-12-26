from __future__ import annotations

import inspect, itertools, os

from typing import Callable, Dict, List, Optional, Tuple, Union

import jijmodeling as jm
import jijzept as jz
import numpy as np
import pandas as pd

from jijmodeling.exceptions import DataError
from jijmodeling.type_annotations import PH_VALUES_INTERFACE

from jijbench.exceptions import ConcurrentFailedError, LoadFailedError
from jijbench.experiment.experiment import Experiment
from jijbench.evaluation.evaluation import Evaluator
from jijbench.benchmark import validation
from jijbench.components import ID, Artifact, ExperimentResultDefaultDir, Table
from jijbench.solver import DefaultSolver

__all__ = []


class Benchmark:
    """Define benchmark.

    Args:
        params (Dict[str, List]): Parameters to be swept in the benchmark. key is parameter name. list is list of value of a parameter.
        solver (str | Callable | List[str | Callable]): solver name or callable solver method. You can set multiple solvers by using list. Accepted `str` type solver names are `{SASampler, SQASampler, JijSASampler, JijSQASampler, JijSwapMovingSampler}`.
            See [jijzept.sampler](https://www.ref.documentation.jijzept.com/reference/sampler/) for more infomation about solvers.
            The callables allow for flexible benchmark.
            If parameter name(`str`) of `params` exists in argument names of callable, it is passed to them.
            However, if passing mathematical models and instance data to callables, `problem` and` instance_data` is also able to use.

        problem (jm.Problem | List[jm.Problem] | None, optional): Mathematical model defined by JijModeling or list of them. Defaults to None.
            See [jijmodeling's reference](https://www.ref.jijmodeling.jijzept.com/) for more information about JijModeling.

        instance_data (PH_VALUES_INTERFACE | List[PH_VALUES_INTERFACE] | List[List[PH_VALUES_INTERFACE]] | Tuple[str, PH_VALUES_INTERFACE] | List[Tuple[str, PH_VALUES_INTERFACE]] | List[List[Tuple[str, PH_VALUES_INTERFACE]]] | None, optional): Instance data for `problem`. Defaults to None.
            If `instance_data` represents a single instance data, one can use:
            - PH_VALUES_INTERFACE (see https://www.documentation.jijzept.com/tutorial/binary_ilp about PH_VALUES_INTERFACE);
            - Tuple[str, PH_VALUES_INTERFACE], which `str` means instance data name.
            If `instance_data` represents multiple instance data, one can use:
            - List[PH_VALUES_INTERFACE], in this case, `problme` is a single;
            - List[List[PH_VALUES_INTERFACE]], in this case, `problem` is multiple;
            - List[Tuple[str, PH_VALUES_INTERFACE]], in this case `problem is a single`;
            - List[List[Tuple[str, PH_VALUES_INTERFACE]]], in this case `problem is multiple`.

        solver_return_name (Dict[str, List[str]] | None, optional): Name solver return values for callable `solver`. Defaults to None.
            Name solver return values for callable `solver`.
            These are used columns of `table`(pd.DataFrame) attribute.
        benchmark_id (int | str | None, optional): ID to distinguish different benchmarks. Defaults to None.
        id_rule (str | Dict[str, str], optional): Rule to automatically assign IDs. Defaults to "uuid".
        save_dir (str, optional): Directory to save benchmark results. Defalt to ./jb_results.
        jijzept_config (str | None, optional): If it is used JijZept by `solver`, this needs. Defaults to None.
        dwave_config (str | None, optional): If it is used Dwave by `solver`, this needs. Defaults to None.

    Attributes:
        table (pandas.DataFrame):
            DataFrame to store benchmark result. This save with csv.
        artifact (Dict):
            Dictionary to store benchmark result. This save python objects that cat't save by csv with pickle
        experiments (List[Experiment]):
            List of Experiment. A experiment id is issued for each combination of `params`.
    """

    def __init__(
        self,
        params: Dict[str, List],
        solver: Union[str, Callable, List[Union[str, Callable]]],
        *,
        problem: Optional[Union[jm.Problem, List[jm.Problem]]] = None,
        instance_data: Optional[
            Union[
                PH_VALUES_INTERFACE,
                List[PH_VALUES_INTERFACE],
                List[List[PH_VALUES_INTERFACE]],
                Tuple[str, PH_VALUES_INTERFACE],
                List[Tuple[str, PH_VALUES_INTERFACE]],
                List[List[Tuple[str, PH_VALUES_INTERFACE]]],
            ]
        ] = None,
        solver_return_name: Optional[Dict[str, List[str]]] = None,
        benchmark_id: Optional[Union[int, str]] = None,
        id_rule: Union[str, Dict[str, str]] = "uuid",
        save_dir: str = ExperimentResultDefaultDir,
        jijzept_config: Optional[str] = None,
        dwave_config: Optional[str] = None,
    ):
        self.params = params
        self.solver_return_name = solver_return_name
        self.id_rules = id_rule
        self.save_dir = save_dir

        self._set_solver(solver)
        self._set_problem(problem)
        self._set_instance_data(instance_data)
        self._id = ID(benchmark_id=benchmark_id)
        self._experiments: List[Experiment] = []
        self._table = Table()
        self._artifact = Artifact()

        if solver_return_name is not None:
            self.name_solver_ret(solver_return_name)

        DefaultSolver.jijzept_config = jijzept_config
        DefaultSolver.dwave_config = dwave_config

    @property
    def solver(self):
        return self._solver

    @validation.on_solver
    def _set_solver(self, solver):
        self._solver = solver

    @property
    def problem(self):
        return self._problem

    @validation.on_problem
    def _set_problem(self, problem):
        self._problem = problem

    @property
    def instance_data(self):
        return self._instance_data

    @validation.on_instance_data
    def _set_instance_data(self, instance_data):
        self._instance_data = instance_data

    @property
    def id(self):
        self._id.benchmark_id

    @property
    def experiments(self):
        return self._experiments

    @property
    def table(self):
        return self._table.data

    @property
    def artifact(self):
        return self._artifact.data

    def run(self, sync=True):
        """run benchmark

        Args:
            sync (bool, optional): True -> sync mode, False -> async mode. Defaults to True. Note that sync=False is not supported using your custom solver.
        """
        if sync is False:
            for solver in self.solver:
                if solver.is_jijzept_sampler is False:
                    raise ConcurrentFailedError(
                        "sync=False is not supported using your custom solver."
                    )
        if self._problem is None:
            problem = [None]
        else:
            problem = self._problem

        if self._instance_data is None:
            instance_data = [[None]]
        else:
            instance_data = self._instance_data

        for solver in self.solver:
            for problem_i, instance_data_i in zip(problem, instance_data):
                for instance_data_ij in instance_data_i:
                    if sync:
                        self._run_by_sync(solver, problem_i, instance_data_ij)
                    else:
                        if "openjij" in solver.name:
                            self._run_by_sync(solver, problem_i, instance_data_ij)
                        else:
                            self._run_by_async(solver, problem_i, instance_data_ij)

    def _run_by_async(self, solver, problem, instance_data, **kwargs):
        _, ph_value = instance_data
        idx = [
            s in inspect.getsource(solver.function)
            for s in DefaultSolver.jijzept_sampler_names
        ]
        args_map = {}
        if any(idx):
            for i_name, b in enumerate(idx):
                if b:
                    name = DefaultSolver.jijzept_sampler_names[i_name]
                    sampler = getattr(jz.sampler, name)(
                        config=DefaultSolver.jijzept_config
                    )
                    parameters = inspect.signature(sampler.sample_model).parameters
                    for i, r in enumerate(itertools.product(*self.params.values())):
                        args = dict([(k, v) for k, v in zip(self.params.keys(), r)])
                        solver_args = {k: w for k, w in args.items() if k in parameters}
                        solution_id = sampler.sample_model(
                            problem, ph_value, sync=False, **solver_args
                        ).solution_id
                        args_map[(i, solution_id)] = args

                    experiment = Experiment(
                        benchmark_id=self._id.benchmark_id, save_dir=self.save_dir
                    )
                    for (i, solution_id), args in args_map.items():
                        solver_args, record = self._setup_experiment(
                            solver, problem, instance_data, False
                        )
                        solver_args.update({"solution_id": solution_id})
                        while True:
                            try:
                                ret = solver(**solver_args)
                                if "APIStatus.SUCCESS" in str(ret):
                                    with experiment:
                                        ret = solver.to_named_ret(ret)
                                        record.update(
                                            args | solver_args | {"i": i} | ret
                                        )
                                        del record["problem"], record["instance_data"]
                                        experiment.store(record)
                                    break
                            except DataError:
                                pass
                    experiment.table.sort_values("i", inplace=True)
                    experiment.table.drop("i", axis=1, inplace=True)

            self._table.data = pd.concat([self._table.data, experiment.table])
            self._table.data.reset_index(drop=True, inplace=True)
            self._artifact.data.update(experiment.artifact)
            self._experiments.append(experiment)

    def _run_by_sync(self, solver, problem, instance_data):
        experiment = Experiment(
            benchmark_id=self._id.benchmark_id, save_dir=self.save_dir
        )
        solver_args, record = self._setup_experiment(
            solver, problem, instance_data, True
        )
        for r in itertools.product(*self.params.values()):
            with experiment:
                solver_args.update(
                    dict([(k, v) for k, v in zip(self.params.keys(), r)])
                )
                ret = solver(**solver_args)
                ret = solver.to_named_ret(ret)
                solver_args.update(ret)
                record.update(dict([(k, v) for k, v in zip(self.params.keys(), r)]))
                record.update(ret)
                experiment.store(record)
        self._table.data = pd.concat([self._table.data, experiment.table])
        self._table.data.reset_index(drop=True, inplace=True)
        self._artifact.data.update(experiment.artifact)
        self._experiments.append(experiment)

    @staticmethod
    def _setup_experiment(solver, problem, instance_data, sync):
        if problem is None:
            problem_name = np.nan
        else:
            problem_name = problem.name

        if instance_data is None:
            instance_data_name, ph_value = np.nan, np.nan
            opt_value = np.nan
        else:
            instance_data_name, ph_value = instance_data
            opt_value = ph_value["opt_value"] if "opt_value" in ph_value else np.nan

        solver_args = {"problem": problem, "instance_data": ph_value, "sync": sync}
        record = {
            "problem_name": problem_name,
            "instance_data_name": instance_data_name,
            "opt_value": opt_value,
            "solver": solver.name,
        }
        return solver_args, record

    def compare(self, key, values=None):
        """Row-by-Row comparison

        Args:
            key (_type_): columns of table.
            values (_type_, optional): values to be compared . Defaults to None.

        Returns:
            pandas.DataFrame: self._table.data.pivot(columns=key, values=values)
        """
        return self._table.data.pivot(columns=key, values=values)

    def evaluate(self, opt_value=None, pr=0.99, expand=True):
        """_summary_

        Args:
            opt_value (_type_, optional): _description_. Defaults to None.
            pr (float, optional): _description_. Defaults to 0.99.
            expand (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        table = Table()
        metrics = pd.DataFrame()
        for experiment in self._experiments:
            evaluator = Evaluator(experiment)
            opt_value = (
                experiment.table["opt_value"][0] if opt_value is None else opt_value
            )
            metrics = pd.concat(
                [
                    metrics,
                    evaluator.calc_typical_metrics(
                        opt_value=opt_value, pr=pr, expand=expand
                    ),
                ]
            )
            table.data = pd.concat([table.data, experiment.table])
        table.data.reset_index(drop=True, inplace=True)
        metrics.reset_index(drop=True, inplace=True)
        self._table = table
        return metrics

    def name_solver_ret(self, ret_names: Dict[str, List[str]]):
        new_ret_names = {}
        for k, v in ret_names.items():
            if k in dir(DefaultSolver):
                new_ret_names[getattr(DefaultSolver(), k).__name__] = v
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
        """_summary_

        Args:
            benchmark_id (Union[int, str]): _description_
            experiment_id (Union[int, str, List[Union[int, str]]], optional): _description_. Defaults to None.
            autosave (bool, optional): _description_. Defaults to True.
            save_dir (str, optional): _description_. Defaults to ExperimentResultDefaultDir.

        Returns:
            _type_: _description_
        """
        experiments = []
        table = Table()
        artifact = Artifact()
        experiment_ids = (
            experiment_id
            if experiment_id
            else get_experiment_id_list(benchmark_id, save_dir)
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
            artifact.data.update(experiment.artifact)

        bench = cls([], lambda: (), benchmark_id=benchmark_id)
        bench._experiments = experiments
        bench._table = table
        bench._artifact = artifact
        return bench


def get_experiment_id_list(
    benchmark_id: Union[int, str],
    save_dir: str = ExperimentResultDefaultDir,
):
    if not f"benchmark_{benchmark_id}" in os.listdir(save_dir):
        msg = f"benchmark_id={benchmark_id} file does not exist in {save_dir}. Please check you benchmark_id."
        raise LoadFailedError(msg)

    return os.listdir(os.path.normcase(f"{save_dir}/benchmark_{benchmark_id}"))


def load(
    benchmark_id: Union[int, str],
    experiment_id: Union[int, str, List[Union[int, str]]] = None,
    autosave: bool = True,
    save_dir: str = ExperimentResultDefaultDir,
):
    return Benchmark.load(
        benchmark_id=benchmark_id,
        experiment_id=experiment_id,
        autosave=autosave,
        save_dir=save_dir,
    )
