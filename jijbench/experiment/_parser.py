from __future__ import annotations

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

if TYPE_CHECKING:
    from dimod import SampleSet
    from jijmodeling import DecodedSamples

    from jijbench.experiment.experiment import Experiment


def _parse_response(
    experiment: "Experiment", response: "SampleSet"
) -> Tuple[List[str], List]:
    table = experiment._table
    info = response.info

    energies = info["energies"]
    num_occurrences = response.record.num_occurrences
    num_reads = info["num_reads"]
    num_sweeps = info["num_sweeps"]
    solving_time = info["solving_time"]
    total_time = info["total_time"]

    num_feasible = np.nan
    num_samples = np.nan

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
        total_time,
        solving_time,
    ]

    return columns, values


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
    num_reads = len(energies)
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


def _parse_jm_problem_decodedsamples(
    experiment: "Experiment", decoded: "DecodedSamples"
) -> Tuple[List[str], List]:
    """extract table data from jijmodeling.DecodedSamples

    This method is called in `Experiment._prase_record`.

    Args:
        experiment (Experiment): experiment object
        decoded (jijmodeling.DecodedSamples): dimod sampleset

    Returns:
        Tuple[List[str], List]: (columns, values)
    """

    table = experiment._table
    energies = decoded.energies
    objectives: np.ndarray = decoded.objectives
    num_occurances = np.nan
    num_reads = np.nan
    num_sweeps = np.nan
    num_feasible = len(decoded.feasibles())
    num_samples = len(decoded.data)
    constraint_violations = {}
    for violation in decoded.constraint_violations:
        for const_name, v in violation.items():
            if const_name in constraint_violations.keys():
                constraint_violations[const_name].append(v)
            else:
                constraint_violations[const_name] = [v]

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
        num_occurances,
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
