from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from jijbench.solver import CallableSolver, DefaultSolver

__all__ = []
