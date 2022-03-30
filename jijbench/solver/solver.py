import inspect
import openjij as oj
import jijzept as jz
from typing import Callable


class CallableSolver:
    def __init__(self, solver):
        self.function = self._parse_solver(solver)
        self._name = self.function.__name__
        self._ret_names = (
            ["response", "decoded"] if solver in dir(DefaultSolver) else None
        )

    def __call__(self, **kwargs):
        parameters = inspect.signature(self.function).parameters
        kwargs = (
            kwargs
            if "kwargs" in parameters
            else {k: v for k, v in kwargs.items() if k in parameters}
        )
        return self.function(**kwargs)

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
            return getattr(DefaultSolver(), solver)
        elif isinstance(solver, Callable):
            return solver
        else:
            return


class DefaultSolver:
    jijzept_config = None
    dwave_config = None
    openjij_sampler_names = ["SASampler", "SQASampler"]
    jijzept_sampler_names = ["JijSASampler"]

    @property
    def SASampler(self):
        return self.openjij_sa_sampler_sample

    @property
    def SQASampler(self):
        return self.openjij_sqa_sampler_sample

    @property
    def JijSASampler(self):
        return self.jijzept_sa_sampler_sample_model

    @classmethod
    def openjij_sa_sampler_sample(cls, problem, ph_value, feed_dict=None, **kwargs):
        return cls._sample_by_openjij(
            oj.SASampler, problem, ph_value, feed_dict, **kwargs
        )

    @classmethod
    def openjij_sqa_sampler_sample(cls, problem, ph_value, feed_dict=None, **kwargs):
        return cls._sample_by_openjij(
            oj.SQASampler, problem, ph_value, feed_dict, **kwargs
        )

    @classmethod
    def jijzept_sa_sampler_sample_model(cls, problem, ph_value, **kwargs):
        return cls._sample_by_jijzept(
            jz.JijSASampler,
            problem,
            ph_value,
            **kwargs,
        )

    @staticmethod
    def _sample_by_openjij(sampler, problem, ph_value, feed_dict, **kwargs):
        if feed_dict is None:
            feed_dict = {const_name: 5.0 for const_name in problem.constraints}
        parameters = inspect.signature(sampler).parameters
        kwargs = {k: w for k, w in kwargs.items() if k in parameters}
        bqm = problem.to_pyqubo(ph_value).compile().to_bqm(feed_dict=feed_dict)
        response = sampler(**kwargs).sample(bqm)
        decoded = problem.decode(response, ph_value=ph_value)
        return response, decoded

    @staticmethod
    def _sample_by_jijzept(sampler, problem, ph_value, sync=True, **kwargs):
        sampler = sampler(config=DefaultSolver.jijzept_config)
        if sync:
            parameters = inspect.signature(sampler.sample_model).parameters
            kwargs = {k: w for k, w in kwargs.items() if k in parameters}
            response = sampler.sample_model(problem, ph_value, sync=sync, **kwargs)
        else:
            response = sampler.get_result(solution_id=kwargs["solution_id"])
        decoded = problem.decode(response, ph_value=ph_value)
        return response, decoded
