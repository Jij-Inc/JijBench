from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from jijbench.benchmark.benchmark import Benchmark
from jijbench.benchmark.validation import on_solver, on_problem, on_instance_data

__all__ = []
