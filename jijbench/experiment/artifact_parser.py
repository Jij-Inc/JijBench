import numpy as np
from typing import TYPE_CHECKING, List, Tuple

if TYPE_CHECKING:
    from jijbench.experiment.experiment import Experiment
    from dimod import SampleSet
    from jijmodeling import DecodedSamples


def get_dimod_sampleset_items(experiment: 'Experiment', response: 'SampleSet') -> Tuple[List[str], List]:
    """extract table data from dimod.SampleSet

    This method is called in `Experiment._reconstruct_record`.

    Args:
        experiment (Experiment): experiment object
        response (dimod.SampleSet): dimod sampleset

    Returns:
        Tuple[List[str], List]: (columns, values)
    """
    energies: np.ndarray = response.record.energy
    num_occurrences = response.record.num_occurrences
    columns = experiment._table.get_energy_columns() + experiment._table.get_num_columns()
    values = [
        list(energies),
        energies.min(),
        energies.mean(),
        energies.std(),
        list(num_occurrences),
        np.nan,
        np.nan,
    ]
    return columns, values


def get_jm_problem_decodedsamples_items(experiment: 'Experiment', decoded: 'DecodedSamples') -> Tuple[List[str], List]:
    """extract table data from jijmodeling.DecodedSamples 

    This method is called in `Experiment._reconstruct_record`.

    Args:
        experiment (Experiment): experiment object
        decoded (jijmodeling.DecodedSamples): dimod sampleset

    Returns:
        Tuple[List[str], List]: (columns, values)
    """

    energies: np.ndarray = decoded.energies
    objectives: np.ndarray = decoded.objectives
    constraint_violations = {}
    for violation in decoded.constraint_violations:
        for const_name, v in violation.items():
            if const_name in constraint_violations.keys():
                constraint_violations[const_name].append(v)
            else:
                constraint_violations[const_name] = [v]
    columns = experiment._table.get_energy_columns()
    columns += experiment._table.get_objective_columns()
    columns += experiment._table.get_num_columns()
    values = [
        list(energies),
        energies.min(),
        energies.mean(),
        energies.std(),
        list(objectives),
        objectives.min(),
        objectives.mean(),
        objectives.std(),
        np.nan,
        len(decoded.feasibles()),
        len(decoded.data),
    ]
    
    for const_name, v in constraint_violations.items():
        v = np.array(v)
        columns += experiment._table.rename_violation_columns(const_name)
        values += [
            list(v),
            v.min(),
            v.mean(),
            v.std(),
        ]
    return columns, values
