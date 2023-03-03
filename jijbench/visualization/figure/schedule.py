from __future__ import annotations

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from jijbench.visualization.figure.interface import Figure

# TODO: docstringの追加


class Schedule(Figure):
    """Visualize schedule.

    Attributes:
        data (OrderedDict):the dict of schedule data. the key is task label, and the value is tuple of three lists. Each is the list of worker, the start time of work, the time length of work.
        fig_ax (tuple[matplotlib.figure.Figure, matplotlib.axes.Subplot]): Figure and Axes of matplotlib. Available after show method is called.
    Example:
        The code below plots a work shift schedule,
        where worker1 works on task0 for 5 unit time from time 3
        and worker2 works on task0 for 6 unit time from time 4.
        The style of the graph (e.g. color) can be changed by arguments of the `.show` method.

        ```python
        >>> from jijbench.figure.schedule import Schedule

        >>> schedule = Schedule()
        >>> schedule.add_data(task_label="task0",
        >>>     workers=[1, 2],
        >>>     start_times=[3, 4],
        >>>     time_lengths=[5, 6],
        >>> )

        >>> schedule.show(color_list=["red"])
        ```
    """

    def __init__(
        self,
    ) -> None:
        self.data = OrderedDict()
        self._fig_ax = None
        self._workers_set = set()

    def add_data(
        self,
        task_label: str,
        workers: list[int] | npt.NDArray,
        start_times: list[int | float] | npt.NDArray,
        time_lengths: list[int | float] | npt.NDArray,
    ) -> None:
        """Add schedule data to data attribute for plot.

        Args:
            task_label (str): the label of the task.
            workers (list[int] | npt.NDArray): the list of worker indices. the length of the list is the number of tasks belonging to task_label.
            start_times (list[int | float] | npt.NDArray): the list of the start time of work. the length of the list is the number of tasks belonging to task_label.
            time_lengths (list[int | float] | npt.NDArray): the list of the time length of work. the length of the list is the number of tasks belonging to task_label.
        """
        if isinstance(workers, np.ndarray):
            workers = workers.tolist()
        if isinstance(start_times, np.ndarray):
            start_times = start_times.tolist()
        if isinstance(time_lengths, np.ndarray):
            time_lengths = time_lengths.tolist()

        if len(workers) != len(start_times):
            raise ValueError("workers and start_times must be the same length.")
        if len(workers) != len(time_lengths):
            raise ValueError("workers and time_lengths must be the same length.")
        self.data.update([(task_label, (workers, start_times, time_lengths))])
        self._workers_set |= set(workers)

    def show(
        self,
        figsize: tuple[int | float] | None = None,
        title: str | None = None,
        color_list: list | None = None,
        alpha_list: list[float] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xticks: list[int | float] | None = None,
        yticks: list[int | float] | None = None,
    ):
        """Plot schedule data which you passed to the add_data method.

        The arguments of the show method are passed to the plot of matplotlib.

        Args:
            figsize (tuple[int | float] | None): the size of figure. The default uses matplotlib's default value.
            title (str | None): the title of figure. Defaults to "time series".
            color_list (list | None): the list of plot line color. The default uses matplotlib's default value.
            alpha_list (list[float] | None): the list of plot line transparency. The default is 1.0 for each plot line.
            xlabel (str | None): the xlabel of figure. Defaults to None.
            ylabel (str | None): the ylabel of figure. Defaults to None.
            xticks (list[int | float] | None): the xticks of figure. The default uses matplotlib's default.
            yticks (list[int | float] | None): the yticks of figure. The default uses matplotlib's default.
        """

        data = self.data

        if len(data) == 0:
            raise RuntimeError(
                "no plot data. Add at least one plot data with add_data method."
            )

        if title is None:
            title = "schedule"

        if (color_list is not None) and (len(color_list) != len(data)):
            raise ValueError("color_list and data must be same length.")

        if alpha_list is None:
            alpha_list = [0.5] * len(data)
        elif len(alpha_list) != len(data):
            raise ValueError("alpha_list and data must be same length.")

        if xlabel is None:
            xlabel = "time"

        if ylabel is None:
            ylabel = "worker"

        if yticks is None:
            yticks = self.workers

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title)

        for i, (task_label, plot_data) in enumerate(data.items()):
            workers, start_times, time_lengths = plot_data
            color = None if color_list is None else color_list[i]
            ax.barh(
                workers,
                time_lengths,
                left=start_times,
                height=0.5,
                label=task_label,
                color=color,
                alpha=alpha_list[i],
            )
            centers = np.array(start_times) + np.array(time_lengths) / 2

            for battery, center, time_length in zip(workers, centers, time_lengths):
                ax.text(
                    center,
                    battery,
                    str(int(time_length)),
                    ha="center",
                    va="center",
                )

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        if xticks is not None:
            ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.legend(
            ncol=len(data),
            bbox_to_anchor=(0, 0.99),
            loc="lower left",
        )

        self._fig_ax = (fig, ax)

    @property
    def fig_ax(self):
        if self._fig_ax is None:
            raise AttributeError(
                "fig_ax attribute is available after show method is called."
            )
        else:
            return self._fig_ax

    @property
    def workers(self):
        return sorted(self._workers_set)
