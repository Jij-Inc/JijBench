from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
from typing import Callable

from jijbench.experiment.experiment import Experiment
from jijbench.visualize.metrics.utils import create_fig_title_list


def get_violations_dict(x: pd.Series) -> dict:
    constraint_violations_indices = x.index[x.index.str.contains("violations")]
    return {index: x[index] for index in constraint_violations_indices}


# boxplotの基本的な機能から徐々に作っていくche


class MetricsPlot:
    def __init__(self, result: Experiment) -> None:
        self.result = result

    def boxplot(
        self,
        f: Callable,
        figsize: tuple[int | float] | None = None,
        title: str | list[str] | None = None,
    ):
        metrics = self.result.table.apply(f, axis=1)
        title_list = create_fig_title_list(metrics, title)

        fig_ax_list = []
        for i, (indices, data) in enumerate(metrics.items()):
            fig, ax = plt.subplots(figsize=figsize)
            print(title_list[i])
            fig.suptitle(title_list[i])
            ax.boxplot(data.values())
            fig_ax_list.append((fig, ax))

        return tuple(fig_ax_list)


"""
    def boxplot_violations(self):
        fig_ax_tuple = self.boxplot(f=get_violations_dict)
        return fig_ax_tuple
"""
