from __future__ import annotations

import jijmodeling as jm
import pandas as pd

from jijbench.experiment.experiment import Experiment
from jijbench.solver.base import Return
from jijbench.functions.factory import RecordFactory


def construct_experiment_from_sampleset(sampleset: jm.SampleSet) -> Experiment:
    """Construct JijBenchmark Experiment instance from a `jm.SampleSet`.

    The visualization function of JijBenchmark is implemented for `jb.Experiment`.
    These function can be applied to the user's `jm.SampleSet` through this function.

    Args:
        sampleset (jm.SampleSet): a JijModeling SampleSet.

    Returns:
        Experiment: a JijBenchmark Experiment instance.
    """
    experiment = Experiment(autosave=False)
    factory = RecordFactory()
    ret = [Return(data=sampleset, name="")]
    record = factory(ret)
    experiment.append(record)
    return experiment


def create_fig_title_list(
    metrics: pd.Series,
    title: str | list[str] | None,
) -> list[str]:
    """Create figure title list for Visualization, each title is passed to `matplotlib.pyplot.suptitle`.

    JijBenchmark`s metrics plot draws a figure for each run (ie each row of `jb.Experiment.table`).
    This function produces a list of length equal to the number of rows in the metrics series (which is expected to be equal to the number of runs),
    each element of the list being the suptitle of the figure.

    Args:
        metrics (pd.Series): A `pd.Series` instance of the metrics for each run.
        title (str | list[str] | None): A title, or a `list` of titles. If `None`, the title list is automatically generated from the metrics indices.

    Returns:
        list[str]: a list of the suptitle of the fugire. Its length is equal to the number of rows in the metrics series.
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
