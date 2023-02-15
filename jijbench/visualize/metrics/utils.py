from __future__ import annotations

import jijmodeling as jm
import pandas as pd

from jijbench.experiment.experiment import Experiment
from jijbench.solver.base import Return
from jijbench.functions.factory import RecordFactory


def construct_experiment_from_sampleset(sampleset: jm.SampleSet) -> Experiment:
    experiment = Experiment(autosave=False)
    factory = RecordFactory()
    ret = [Return(data=sampleset, name="")]
    record = factory(ret)
    experiment.append(record)
    return experiment


def create_fig_title_list(
    metrics: pd.Series,
    title: str | list[str] | None,
):
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
