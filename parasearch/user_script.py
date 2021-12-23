from jijmodeling import Problem
from jijmodeling.decode import DecodedSamples
from typing import Dict


def transpile_problem(problem: Problem) -> Problem:
    return problem


def make_initial_multipliers(problem: Problem) -> Dict[str, float]:
    multipliers = {}
    for key in problem.constraints.keys():
        multipliers[key] = 1
    return multipliers


def parameter_update(problem: Problem,
                     decoded: DecodedSamples,
                     multipliers: Dict[str, float]) -> Dict[str, float]:
    next_multipliers = {}
    for key, value in decoded.constraint_violations[0].items():
        if value > 0:
            next_multipliers[key] = 5 * multipliers[key]
        else:
            next_multipliers[key] = multipliers[key]
    return next_multipliers
