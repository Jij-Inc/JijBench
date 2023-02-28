from __future__ import annotations

from matplotlib import axes, figure

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import pandas as pd
import re

import plotly.express as px
from typing import Callable, cast

import jijbench as jb
from jijbench.visualize.metrics.utils import (
    create_fig_title_list,
    is_multipliers_column_valid,
)


def get_violations_dict(x: pd.Series) -> dict:
    """Get a dictionary of constraint violations from `pd.Series`.

    This function is intended to be used for `jb.Experiment.table` as example.

    Args:
        x (pd.Series): a Series of benchmark result. Expected to include information on constraint violations.

    Returns:
        dict: a dictionary of constraint violations

    Example:
        The code below get the dictionary of constraint violations for each row in experiment.

        ```python
        import jijbench as jb
        import jijzept as jz
        from jijbench.visualize.metrics.plot import get_violations_dict

        problem = jb.get_problem("TSP")
        instance_data = jb.get_instance_data("TSP")[0][1]
        multipliers1 = {"onehot_time": 0.003, "onehot_location": 0.003}
        multipliers2 = {"onehot_time": 0.3, "onehot_location": 0.3}

        config_path = "XX"
        sa_sampler = jz.JijSASampler(config=config_path)

        bench = jb.Benchmark(
            params = {
                "model": [problem],
                "feed_dict": [instance_data],
                "multipliers": [multipliers1, multipliers2],
            },
            solver = [sa_sampler.sample_model],
        )
        experiment = bench()
        metrics = experiment.table.apply(get_violations_dict, axis=1)
        ```
    """
    constraint_violations_indices = x.index[x.index.str.contains("violations")]
    return {index: x[index] for index in constraint_violations_indices}


def calc_samplemean_from_array(x: pd.Series, column_name: str) -> float:
    num_occ = x["num_occurrences"]
    array = x[column_name]
    mean = np.sum(num_occ * array) / np.sum(num_occ)
    return mean


def get_multiplier(x: pd.Series, constraint_name: str) -> float:
    multipliers = x["multipliers"]
    return multipliers[constraint_name]


class MetricsPlot:
    def __init__(self, result: jb.Experiment) -> None:
        """Visualize the metrics of a benchmark result.

         Attributes:
             result (jb.Experiment): a benchmark result.

        Example:
             Below is the code to boxplot the constraint violations.
             Check the docstring of each method for details.

            ```python
            import jijbench as jb
            import jijzept as jz
            from jijzept.sampler.openjij.sa_cpu import JijSAParameters
            from jijbench.visualize.metrics.plot import MetricsPlot

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]
            multipliers1 = {"onehot_time": 0.003, "onehot_location": 0.003}
            multipliers2 = {"onehot_time": 0.3, "onehot_location": 0.3}

            config_path = "XX"
            sa_parameter = JijSAParameters(num_reads=15)
            sa_sampler = jz.JijSASampler(config=config_path)

            bench = jb.Benchmark(
                params = {
                    "model": [problem],
                    "feed_dict": [instance_data],
                    "parameters": [sa_parameter],
                    "multipliers": [multipliers1, multipliers2],
                },
                solver = [sa_sampler.sample_model],
            )
            result = bench()

            mplot = MetricsPlot(result)
            fig_ax_tuple = mplot.boxplot_violations()
        ```
        """
        self.result = result

    def boxplot(
        self,
        f: Callable,
        figsize: tuple[int | float] | None = None,
        title: str | list[str] | None = None,
        title_fontsize: float | None = None,
        xticklabels_size: float | None = None,
        xticklabels_rotation: float | None = None,
        ylabel: str | None = None,
        ylabel_size: float | None = None,
        yticks: list[int | float] | None = None,
        **matplotlib_boxplot_kwargs,
    ):
        """Draw a box and whisker plot of the metrics based on `result` data using matplotlib.boxplot.

        This method applies the function f to the result (i.e. `jb.Experiment`) to get the metrics (pd.Series), and draw boxplot of this metrics.
        This metrics series calculated as metrics = self.result.table.apply(f, axis=1) in this method assumes the following structure.
            The length is equal to the number of rows in `result.table`.
            the element is a dictionary where the key is the name of each boxplot and the value is the np.array of the boxplot data.
        This method returns a figure and axes, so you can post-process them to change the appearance of the plot.
        See also the example below.

        Args:
            f (Callable): Callalbe to apply to table and get metrics.
            figsize (tuple[int | float] | None): the size of figure. The default uses matplotlib's default value.
            title (str | list[str] | None): the title of figure. The default uses the indices of `result.table`.
            title_fontsize (float | None): the fontsize of the title.The default uses matplotlib's default value.
            xticklabels_size (float | None): the fontsize of the xticklabels (i.e. the name of each boxplot). The default uses matplotlib's default value.
            xticklabels_rotation (float | None): the rotation angle of the xticklabels in degree.The default uses matplotlib's default value.
            ylabel (str | None): the ylabel of figure. Defaults to None.
            ylabel_size (float | None): the fontsize of the ylabel. The default uses matplotlib's default value.
            yticks (list[int | float] | None): the yticks of figure. Default to only integers by`MaxNLocator(integer=True)`.
            matplotlib_boxplot_kwargs: the parameter passed to matplotlib.boxplot.

        Returns:
            tuple[tuple[matplotlib.figure.Figure, matplotlib.axes.Subplot]]: A tuple of length equal to the number of rows in result. each element of is a tuple of figure and axes.

        Example:
            The code below draws a boxplot of violations of each constraint.
            Note that result (i.e. `jb.Experiment`) holds violations for each constraint in np.array format.
            In the first example, postprocessing the figure and axes changes the appearance of the plot.

            ```python
            import jijbench as jb
            import jijzept as jz
            from jijzept.sampler.openjij.sa_cpu import JijSAParameters
            from jijbench.visualize.metrics.plot import MetricsPlot
            import pandas as pd

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]
            multipliers1 = {"onehot_time": 0.003, "onehot_location": 0.003}
            multipliers2 = {"onehot_time": 0.3, "onehot_location": 0.3}

            config_path = "XX"
            sa_parameter = JijSAParameters(num_reads=15)
            sa_sampler = jz.JijSASampler(config=config_path)

            bench = jb.Benchmark(
                params = {
                    "model": [problem],
                    "feed_dict": [instance_data],
                    "parameters": [sa_parameter],
                    "multipliers": [multipliers1, multipliers2],
                },
                solver = [sa_sampler.sample_model],
            )
            result = bench()

            def get_violations_dict(x: pd.Series) -> dict:
                constraint_violations_indices = x.index[x.index.str.contains("violations")]
                return {index: x[index] for index in constraint_violations_indices}

            mplot = MetricsPlot(result)
            fig_ax_tuple = mplot.boxplot(f=get_violations_dict)

            # you can post-process figure and axes to change the appearance of the plot.
            for fig, ax in fig_ax_tuple:
                fig.suptitle("my title")
                display(fig)
            ```

            By using the `construct_experiment_from_samplesets function`,
            `boxplot` can also be used for `jm.SampleSet` obtained without `JijBenchmark`.

            ```python
            import jijbench as jb
            import jijzept as jz
            from jijbench.visualize.metrics.plot import MetricsPlot
            from jijbench.visualize.metrics.utils import construct_experiment_from_samplesets
            import pandas as pd

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]
            multipliers = {"onehot_time": 0.003, "onehot_location": 0.003}

            config_path = "XX"
            sampler = jz.JijSASampler(config=config_path)
            sampleset = sampler.sample_model(model=problem, feed_dict=instance_data, multipliers=multipliers, num_reads=100)

            def get_violations_dict(x: pd.Series) -> dict:
                constraint_violations_indices = x.index[x.index.str.contains("violations")]
                return {index: x[index] for index in constraint_violations_indices}

            result = construct_experiment_from_samplesets(sampleset)
            mplot = MetricsPlot(result)
            fig_ax_tuple = mplot.boxplot(f=get_violations_dict)
            ```
        """
        metrics = self.result.table.apply(f, axis=1)
        title_list = create_fig_title_list(metrics, title)

        fig_ax_list = []
        for i, (indices, data) in enumerate(metrics.items()):
            fig, ax = plt.subplots(figsize=figsize)
            fig.suptitle(title_list[i], fontsize=title_fontsize)
            ax.boxplot(data.values(), **matplotlib_boxplot_kwargs)
            ax.set_xticklabels(
                data.keys(), size=xticklabels_size, rotation=xticklabels_rotation
            )
            ylabel = cast("str", ylabel)
            ax.set_ylabel(ylabel, size=ylabel_size)
            if yticks is None:
                # make yticks integer only
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
            else:
                ax.set_yticks(yticks)
            fig_ax_list.append((fig, ax))

        return tuple(fig_ax_list)

    def boxplot_violations(
        self,
        figsize: tuple[int | float] | None = None,
        title: str | list[str] | None = None,
        title_fontsize: float | None = None,
        constraint_name_fontsize: float | None = None,
        constraint_name_fontrotation: float | None = None,
        ylabel: str | None = None,
        ylabel_size: float | None = None,
        yticks: list[int | float] | None = None,
        **matplotlib_boxplot_kwargs,
    ) -> tuple[tuple[figure.Figure, axes.Subplot]]:
        """Draw a box and whisker plot of the constraint violations of `result` data using matplotlib.boxplot.

        The arguments are passed to matplotlib functions to change the appearance of the plot.
        matplotlib_boxplot_kwargs are passed to matplotlib.boxplot, and defaults to `{showmeans: True, whis: [0, 100]}`.
            showmeans=True shows mean values in markers.
            the outliers are not considered and whiskers match maxima and minima by whis=[0, 100].
        This method returns a figure and axes, so you can post-process them to change the appearance of the plot.
        See also the example below.

        Args:
            figsize (tuple[int | float] | None): the size of figure. The default uses matplotlib's default value.
            title (str | list[str] | None): the title of figure. The default uses the indices of `result.table`.
            title_fontsize (float | None): the fontsize of the title.The default uses matplotlib's default value.
            constraint_name_size (float | None): the fontsize of the constraint name (i.e. xticklabels). The default uses matplotlib's default value.
            constraint_name_rotation (float | None): the rotation angle of the constraint name in degree.The default uses matplotlib's default value.
            ylabel (str | None): the ylabel of figure. Defaults to "constraint violations".
            ylabel_size (float | None): the fontsize of the ylabel. The default uses matplotlib's default value.
            yticks (list[int | float] | None): the yticks of figure. Default to only integers by`MaxNLocator(integer=True)`.
            matplotlib_boxplot_kwargs: the parameter passed to matplotlib.boxplot. Defaults to `{showmeans: True, whis: [0, 100]}`.

        Returns:
            tuple[tuple[matplotlib.figure.Figure, matplotlib.axes.Subplot]]: A tuple of length equal to the number of rows in result. each element of is a tuple of figure and axes.


        Example:
            Below is the code to boxplot the constraint violations.
            In the first example, postprocessing the figure and axes changes the appearance of the plot.

            ```python
            import jijbench as jb
            import jijzept as jz
            from jijzept.sampler.openjij.sa_cpu import JijSAParameters
            from jijbench.visualize.metrics.plot import MetricsPlot

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]
            multipliers1 = {"onehot_time": 0.003, "onehot_location": 0.003}
            multipliers2 = {"onehot_time": 0.3, "onehot_location": 0.3}

            config_path = "XX"
            sa_parameter = JijSAParameters(num_reads=15)
            sa_sampler = jz.JijSASampler(config=config_path)

            bench = jb.Benchmark(
                params = {
                    "model": [problem],
                    "feed_dict": [instance_data],
                    "parameters": [sa_parameter],
                    "multipliers": [multipliers1, multipliers2],
                },
                solver = [sa_sampler.sample_model],
            )
            result = bench()

            mplot = MetricsPlot(result)
            fig_ax_tuple = mplot.boxplot_violations()

            # you can post-process figure and axes to change the appearance of the plot.
            for fig, ax in fig_ax_tuple:
                fig.suptitle("my title")
                display(fig)
            ```

            By using the `construct_experiment_from_samplesets function`,
            `boxplot_violations` can also be used for `jm.SampleSet` obtained without `JijBenchmark`.

            ```python
            import jijbench as jb
            import jijzept as jz
            from jijbench.visualize.metrics.plot import MetricsPlot
            from jijbench.visualize.metrics.utils import construct_experiment_from_samplesets

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]
            multipliers = {"onehot_time": 0.003, "onehot_location": 0.003}

            config_path = "XX"
            sampler = jz.JijSASampler(config=config_path)
            sampleset = sampler.sample_model(model=problem, feed_dict=instance_data, multipliers=multipliers, search=False, num_reads=100)

            result = construct_experiment_from_samplesets(sampleset)
            mplot = MetricsPlot(result)
            fig_ax_tuple = mplot.boxplot_violations()
            ```
        """
        if ylabel is None:
            ylabel = "constraint violations"
        if len(matplotlib_boxplot_kwargs) == 0:
            # Show the arithmetic means in boxplot.
            matplotlib_boxplot_kwargs["showmeans"] = True
            # Make boxplot whisker positions min and max.
            matplotlib_boxplot_kwargs["whis"] = [0, 100]
        fig_ax_tuple = self.boxplot(
            f=get_violations_dict,
            figsize=figsize,
            title=title,
            title_fontsize=title_fontsize,
            xticklabels_size=constraint_name_fontsize,
            xticklabels_rotation=constraint_name_fontrotation,
            ylabel=ylabel,
            ylabel_size=ylabel_size,
            yticks=yticks,
            **matplotlib_boxplot_kwargs,
        )

        # Add a horizontal line to indicate that the constraint is satisfied. (violation = 0)
        fig_ax_list = []
        for fig, ax in fig_ax_tuple:
            ax.axhline(0, xmin=0, xmax=1, color="gray", linestyle="dotted")
            fig_ax_list.append((fig, ax))
        return tuple(fig_ax_list)

    def parallelplot_experiment(
        self,
        color_column_name: str | None = None,
    ):
        result_table = self.result.table

        # The key is a column name (str), and the value is the data of each column (pd.Series).
        data_to_create_df_parallelplot = {}

        # multiplires (If self.result has a valid multipliers column)
        if is_multipliers_column_valid(result_table):
            for constraint_name in result_table["multipliers"].values[0].keys():
                data_to_create_df_parallelplot[
                    constraint_name + "_multiplier"
                ] = result_table.apply(
                    get_multiplier, axis=1, constraint_name=constraint_name
                )

        # objective
        data_to_create_df_parallelplot["samplemean_objective"] = result_table.apply(
            calc_samplemean_from_array, axis=1, column_name="objective"
        )

        # violations
        for violation_column_name in result_table.columns[
            result_table.columns.str.contains("violations")
        ]:
            data_to_create_df_parallelplot[
                "samplemean_" + violation_column_name
            ] = result_table.apply(
                calc_samplemean_from_array, axis=1, column_name=violation_column_name
            )

        # Extract series about violations from data_to_create_df_parallelplot (key starts with 'samplemean_' and ends with '_violations') and calculates samplemean_total_violations by taking sum.
        start, end = re.compile(r"^samplemean_"), re.compile(r"_violations$")

        violation_series = [
            series
            for name, series in data_to_create_df_parallelplot.items()
            if start.search(name) and end.search(name)
        ]

        if violation_series:
            data_to_create_df_parallelplot["samplemean_total_violations"] = sum(
                violation_series
            )

        self.df_parallelplot = df_parallelplot = pd.DataFrame(
            data_to_create_df_parallelplot
        )

        """
        if color_column_name is None:
            if "samplemean_total_violations" in df_parallelplot.columns:
                color_column_name = "samplemean_total_violations"
            else:
                color_column_name = "samplemean_objective"
        """

        fig = px.parallel_coordinates(
            df_parallelplot.reset_index(),
            color=color_column_name,
        )

        return fig
