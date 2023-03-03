from __future__ import annotations

from jijbench import datasets
from jijmodeling.type_annotations import PH_VALUES_INTERFACE


__all__ = []


def get_instance_data(
    problem_name, size="small"
) -> list[tuple[str, PH_VALUES_INTERFACE]]:
    cls = getattr(datasets, problem_name)()

    instance_data = []
    for name in cls.instance_names(size=size):
        instance_data.append((name, cls.get_instance(size=size, instance_name=name)))

    return instance_data
