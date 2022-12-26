from __future__ import annotations

import math, warnings

from typing import TYPE_CHECKING, List, Tuple

import numpy as np

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


def _parse_jm_sampleset(
    experiment: "Experiment", jm_sampleset: "jm_SampleSet"
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
    energies = np.array(jm_sampleset.evaluation.energy)
    objectives = np.array(jm_sampleset.evaluation.objective)
    num_occurrences = np.array(jm_sampleset.record.num_occurrences)
    num_reads = np.nan
    num_sweeps = np.nan

    constraint_violations = jm_sampleset.evaluation.constraint_violations

    num_feasible = sum(jm_sampleset.feasible().record.num_occurrences)
    num_samples = num_occurrences.sum()

    sampling_time = np.nan
    # TODO スキーマが変わったら変更する
    solving_time = jm_sampleset.measuring_time.solve
    if solving_time.solve is None:
        execution_time = np.nan
        warnings.warn(
            "'solve' of jijmodeling.SampleSet is None. Give it if you want to evaluate automatically."
        )
    else:
        execution_time = solving_time.solve

    columns = table.get_energy_columns()
    columns += table.get_objective_columns()
    columns += table.get_num_columns()
    columns += table.get_time_columns()

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
        sampling_time,
        execution_time,
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
