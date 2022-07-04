from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

from jijbench.problem.get import get_instance_data, get_problem
from jijbench.problem.problem import TSP, TSPTW, BinPacking, Knapsack

__all__ = [
    "BinPacking",
    "Knapsack",
    "TSP",
    "TSPTW",
]
