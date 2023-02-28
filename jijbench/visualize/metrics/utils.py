from __future__ import annotations

import jijmodeling as jm
from numbers import Number
import pandas as pd
import typing as tp

from jijbench.experiment.experiment import Experiment
from jijbench.functions.concat import Concat
from jijbench.solver.base import Parameter, Return
from jijbench.functions.factory import RecordFactory


def construct_experiment_from_samplesets(
    samplesets: list[jm.SampleSet] | jm.SampleSet,
    additional_data: dict[str, list[tp.Any]] | None = None,
) -> Experiment:
    """Construct `jb.Experiment` instance from a list of `jm.SampleSet`.

    The visualization function of JijBenchmark is implemented for `jb.Experiment`.
    These function can be applied to the user's `jm.SampleSet` through this function.

    Args:
        samplesets (list[jm.SampleSet] | jm.SampleSet): a list of JijModeling SampleSet. You can also just give a single `jm.SampleSet`.
        additional_data (dict[str, list[tp.Any]] | None):  a dictionary of data to store in the experiment.
            The key will be the jb.Experiment.table column name and the value is the list of elements stored in the table.
            The length of this list must equal the length of samplesets list.
            Defaults to None.

    Returns:
        Experiment: a `jb.Experiment` instance. The number of rows in `jb.Experiment.table` is equal to the length of samplesets.

    Example:
        The code below solves the TSP problem and gets the jb.Experiment from that sampleset.

        ```python
        import jijbench as jb
        import jijzept as jz
        from jijbench.visualize.metrics.utils import construct_experiment_from_samplesets

        problem = jb.get_problem("TSP")
        instance_data = jb.get_instance_data("TSP")[0][1]

        # config_path = "XX"
        sampler = jz.JijSASampler(config=config_path)

        samplesets = []
        onehot_time_multipliers = []
        onehot_location_multipliers = []

        for onehot_time_multiplier in [0.01, 0.1]:
            for onehot_location_multiplier in [0.01, 0.1]:
                multipliers = {"onehot_time": onehot_time_multiplier, "onehot_location": onehot_location_multiplier}
                sampleset = sampler.sample_model(
                    model=problem,
                    feed_dict=instance_data,
                    multipliers=multipliers,
                    num_reads=10,
                )
                samplesets.append(sampleset)
                onehot_time_multipliers.append(onehot_time_multiplier)
                onehot_location_multipliers.append(onehot_location_multiplier)

        additional_data = {
            "onehot_time_multiplier": onehot_time_multipliers,
            "onehot_location_multiplier": onehot_location_multipliers,
        }
        experiment = construct_experiment_from_samplesets(samplesets, additional_data)
        ```
    """
    if isinstance(samplesets, jm.SampleSet):
        samplesets = [samplesets]

    if additional_data is None:
        additional_data = {}
    else:
        for v in additional_data.values():
            if len(v) != len(samplesets):
                raise TypeError(
                    "The list assigned to the value of additional_data must have the same length as the sampleset."
                )

    # Convert additional_data to JijBenchmark Parameters.
    params = [
        [
            v if isinstance(v, Parameter) else Parameter(v, k)
            for k, v in zip(additional_data.keys(), r)
        ]
        for r in zip(*additional_data.values())
    ]
    experiment = Experiment(autosave=False)
    for i, sampleset in enumerate(samplesets):
        factory = RecordFactory()
        ret = [Return(data=sampleset, name="")]
        record = factory(ret)
        # Concat additional_data if given.
        if len(params) >= 1:
            record = Concat()([RecordFactory()(params[i]), record])
        experiment.append(record)
    return experiment


def create_fig_title_list(
    metrics: pd.Series,
    title: str | list[str] | None,
) -> list[str]:
    """Create figure title list for Visualization, each title is passed to `matplotlib.pyplot.suptitle`.

    This function produces a title list of length equal to the number of rows in the metrics series.
    JijBenchmark`s metrics plot draws a figure for each run (i.e. each row of `jb.Experiment.table`),
    and each element of the returned list is expected to be the suptitle of each figure.

    Args:
        metrics (pd.Series): A `pd.Series` instance of the metrics for each run.
        title (str | list[str] | None): A title, or a `list` of titles. If `None`, the title list is automatically generated from the metrics indices.

    Returns:
        list[str]: a list of the suptitle of the figure. Its length is equal to the number of rows in the metrics series.
    """
    if isinstance(title, list):
        title_list = title
        return title_list
    elif isinstance(title, str):
        title_list = [title for _ in range(len(metrics))]
        return title_list
    elif title is None:
        title_list = []
        index_names = metrics.index.names
        for indices, data in metrics.items():
            if indices is None:
                title_list.append("")
            else:
                # If user don't give title, the title list is automatically generated from the metrics indices.
                title_list.append(
                    "\n".join(
                        [
                            f"{index_name}: {index}"
                            for index_name, index in zip(index_names, indices)
                        ]
                    )
                )
        return title_list
    else:
        raise TypeError("title must be str or list[str].")


def is_multipliers_column_valid(df: pd.DataFrame) -> bool:
    if not "multipliers" in df.columns:
        return False

    def is_multiplier_valid(x: pd.Series, constraint_names: list[str]) -> bool:
        multipliers = x["multipliers"]
        if not isinstance(multipliers, dict):
            return False
        for key, value in multipliers.items():
            if not isinstance(key, str):
                return False
            if not isinstance(value, Number):
                return False
        if list(multipliers.keys()) != constraint_names:
            return False
        return True

    try:
        constraint_names = list(df["multipliers"].values[0].keys())
    except AttributeError:
        return False
    is_multiplier_valid_array = df.apply(
        is_multiplier_valid,
        axis=1,
        constraint_names=constraint_names,
    )
    return is_multiplier_valid_array.values.all()
