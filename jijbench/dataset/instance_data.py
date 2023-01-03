from __future__ import annotations

from typing import List, Tuple

from jijmodeling.type_annotations import PH_VALUES_INTERFACE

from jijbench import dataset

__all__ = []


def get_problem(problem_name):
    return getattr(dataset, problem_name).problem


def get_instance_data(
    problem_name, size="small"
) -> List[Tuple[str, PH_VALUES_INTERFACE]]:
    cls = getattr(dataset, problem_name)()

    instance_data = []
    for name in cls.instance_names(size=size):
        instance_data.append((name, cls.get_instance(size=size, instance_name=name)))

    return instance_data
