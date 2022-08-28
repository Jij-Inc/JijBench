from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import math

if TYPE_CHECKING:
    from dimod import SampleSet
    from jijmodeling.sampleset import SampleSet as jm_SampleSet

    from jijbench.experiment.experiment import Experiment


def _parse_dimod_sampleset(
    experiment: "Experiment", response: "SampleSet"
) -> Tuple[List[str], List]:
    """extract table data from dimod.SampleSet

    This method is called in `Experiment._parse_record`.

    Args:
        experiment (Experiment): experiment object
        response (dimod.SampleSet): dimod sampleset

    Returns:
        Tuple[List[str], List]: (columns, values)
    """
    table = experiment._table
    energies = response.record.energy
    num_occurrences = response.record.num_occurrences
    num_reads = num_occurrences.sum()
    if "schedule" in response.info:
        num_sweeps = response.info["schedule"]["num_sweeps"]
    else:
        num_sweeps = np.nan
    num_feasible = np.nan
    num_samples = np.nan

    if "sampling_time" in response.info:
        sampling_time = response.info["sampling_time"]
    else:
        sampling_time = np.nan
    if "execution_time" in response.info.keys():
        execution_time = response.info["execution_time"]
    else:
        if "sampling_time" in response.info:
            execution_time = response.info["sampling_time"] / num_reads
        else:
            execution_time = np.nan

    columns = table.get_energy_columns()
    columns += table.get_num_columns()
    columns += table.get_time_columns()
    values = [
        energies,
        energies.min(),
        energies.mean(),
        energies.std(),
        num_occurrences,
        num_reads,
        num_sweeps,
        num_feasible,
        num_samples,
        sampling_time,
        execution_time,
    ]

    return columns, values


def _parse_jm_problem_sampleset(
    experiment: "Experiment", decoded: "jm_SampleSet"
) -> Tuple[List[str], List]:
    """extract table data from jijmodeling.SampleSet

    This method is called in `Experiment._parse_record`.

    Args:
        experiment (Experiment): experiment object
        decoded (jijmodeling.SampleSet): jijmodeling sampleset

    Returns:
        Tuple[List[str], List]: (columns, values)
    """

    table = experiment._table
    energies = np.array(decoded.evaluation.energy)
    objectives = np.array(decoded.evaluation.objective)
    num_occurrences = np.array(decoded.record.num_occurrences)
    num_reads = np.nan
    num_sweeps = np.nan

    constraint_violations = decoded.evaluation.constraint_violations

    # TODO: add .feasibles() to jm.SampleSet (https://github.com/Jij-Inc/JijModelingExpression/issues/70) to rewrite
    # TODO: num_feasible = decoded.feasibles().num_occurrences.sum()
    # calculate num_occurrences with feasible solutions
    feasible_num_occurrences = []
    for i, num_occur_value in enumerate(num_occurrences):
        violation = 0
        for _, v in constraint_violations.items():
            violation += v[i]
        if math.isclose(violation, 0):
            feasible_num_occurrences.append(num_occur_value)

    num_feasible = sum(feasible_num_occurrences)

    num_samples = num_occurrences.sum()

    columns = table.get_energy_columns()
    columns += table.get_objective_columns()
    columns += table.get_num_columns()

    values = [
        energies,
        energies.min(),
        energies.mean(),
        energies.std(),
        objectives,
        objectives.min(),
        objectives.mean(),
        objectives.std(),
        num_occurrences,
        num_reads,
        num_sweeps,
        num_feasible,
        num_samples,
    ]

    for const_name, v in constraint_violations.items():
        v = np.array(v)
        columns += table.rename_violation_columns(const_name)
        values += [
            list(v),
            v.min(),
            v.mean(),
            v.std(),
        ]
    return columns, values
