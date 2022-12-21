from __future__ import annotations

import inspect

from typing import Callable, Optional

import jijmodeling as jm
import jijzept as jz

from jijbench.exceptions import SolverFailedError

__all__ = []


class CallableSolver:
    def __init__(self, solver):
        self.function = self._parse_solver(solver)
        self._name = self.function.__name__
        self._ret_names = (
            ["response", "decoded"] if solver in dir(DefaultSolver) else None
        )

    def __call__(self, **kwargs):
        parameters = inspect.signature(self.function).parameters
        is_kwargs = any([p.kind == 4 for p in parameters.values()])
        kwargs = (
            kwargs
            if is_kwargs
            else {k: v for k, v in kwargs.items() if k in parameters}
        )
        try:
            ret = self.function(**kwargs)
        except Exception as e:
            raise SolverFailedError(e)
        return ret

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
    jijzept_config: Optional[str] = None
    dwave_config: Optional[str] = None
    jijzept_sampler_names = ["JijSASampler", "JijSQASampler", "JijSwapMovingSampler"]

    @property
    def JijSASampler(self):
        return self.jijzept_sa_sampler_sample_model

    @property
    def JijSQASampler(self):
        return self.jijzept_sqa_sampler_sample_model

    @property
    def JijSwapMovingSampler(self):
        return self.jijzept_swapmoving_sampler_sample_model

    @classmethod
    def jijzept_sa_sampler_sample_model(cls, problem, instance_data, **kwargs):
        return cls._sample_by_jijzept(
            jz.JijSASampler,
            problem,
            instance_data,
            **kwargs,
        )

    @classmethod
    def jijzept_sqa_sampler_sample_model(cls, problem, instance_data, **kwargs):
        return cls._sample_by_jijzept(
            jz.JijSQASampler, problem, instance_data, **kwargs
        )

    @classmethod
    def jijzept_swapmoving_sampler_sample_model(cls, problem, instance_data, **kwargs):
        return cls._sample_by_jijzept(
            jz.JijSwapMovingSampler, problem, instance_data, **kwargs
        )

    @staticmethod
    def _sample_by_jijzept(sampler, problem, instance_data, sync=True, **kwargs):
        sampler = sampler(config=DefaultSolver.jijzept_config)
        if sync:
            parameters = inspect.signature(sampler.sample_model).parameters
            kwargs = {k: w for k, w in kwargs.items() if k in parameters}
            response: jm.SampleSet = sampler.sample_model(
                problem, instance_data, sync=sync, **kwargs
            )
        else:
            response: jm.SampleSet = jz.response.JijModelingResponse.empty_response(
                jz.response.APIStatus.PENDING, sampler.client, kwargs["solution_id"]
            )
            response.get_result()

        return response
