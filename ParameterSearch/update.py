from jijmodeling import Problem
from openjij import Response
from dimod import SampleSet
from typing import Dict, Union


def make_initial_multipliers(problem: Problem):
    multipliers = {}
    for key in problem.constraints.keys():
        multipliers[key] = 1
    return multipliers


def parameter_update(problem: Problem,
                     response: Union[SampleSet, Response],
                     multipliers: Dict[str, float]):
    next_multipliers = {}
    # for key, value in decoded.constraint_violations[0].items():
    #     if value > 0:
    #         next_multipliers[key] = 5 * multipliers[key]
    #     else:
    #         next_multipliers[key] = multipliers[key]
    return multipliers
#     return next_multipliers
