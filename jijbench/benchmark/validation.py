import functools
from jijbench import JijModelingTarget
from jijbench import problems
from jijbench.solver import CallableSolver


def on_solver(fn):
    @functools.wraps(fn)
    def wrapper(obj, *args, **kwargs):
        (solver,) = args
        if isinstance(solver, (list, tuple)):
            callable_solvers = []
            for s in solver:
                callable_solvers.append(CallableSolver(s))
            if isinstance(solver, tuple):
                callable_solvers = tuple(callable_solvers)
        else:
            callable_solvers = [CallableSolver(solver)]
        fn(obj, callable_solvers)

    return wrapper


def on_target(fn):
    @functools.wraps(fn)
    def wrapper(obj, *args, **kwargs):
        (_targets,) = args
        if isinstance(_targets, (list, tuple)):
            targets = []
            for t in _targets:
                if isinstance(t, JijModelingTarget):
                    targets.append(t)
                if isinstance(t, str):
                    targets.append(getattr(problems, t)())
            if isinstance(_targets, tuple):
                targets = tuple(targets)
        if isinstance(_targets, str):
            targets = [getattr(problems, _targets)()]
        if isinstance(_targets, JijModelingTarget):
            targets = [_targets]
        fn(obj, targets)

    return wrapper
