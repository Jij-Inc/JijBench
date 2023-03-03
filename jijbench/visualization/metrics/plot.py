from __future__ import annotations

from matplotlib import axes, figure

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from numbers import Number
import numpy as np
import pandas as pd
import re

import plotly
import plotly.express as px
from typing import Callable, cast, Literal

import jijbench as jb
from jijbench.exceptions.exceptions import UserFunctionFailedError
from jijbench.visualization.metrics.utils import (
    _create_fig_title_list,
    _df_has_valid_multipliers_column,
    _df_has_number_array_column_target_name,
    _df_has_number_column_target_name,
)

AXIS_LABEL_POS = Literal["top", "bottom"]


def _get_violations_dict(x: pd.Series) -> dict:
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
        from jijbench.visualization.metrics.plot import _get_violations_dict

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
        metrics = experiment.table.apply(_get_violations_dict, axis=1)
        ```
    """
    constraint_violations_indices = x.index[x.index.str.contains("violations")]
    return {index: x[index] for index in constraint_violations_indices}


def _calc_mean_of_array(x: pd.Series, column_name: str) -> float:
    """
    Gets the mean of the elements in the specified column ,assuming this element is an np.ndarray.

    This function is intended to be used for `jb.Experiment.table`
    as experiment.table.apply(_calc_mean_of_array, axis=1, column_name=column_name)
    """
    num_occ = x["num_occurrences"]
    array = x[column_name]
    mean = np.sum(num_occ * array) / np.sum(num_occ)
    return mean


def _get_multiplier(x: pd.Series, constraint_name: str) -> float:
    """
    Gets the multiplier of given constraint name from `pd.Series`.

    This function is intended to be used for `jb.Experiment.table`
    as experiment.table.apply(_get_multiplier, axis=1, constraint_name=constraint_name)
    """
    multipliers = x["multipliers"]
    return multipliers[constraint_name]


class MetricsPlot:
    def __init__(self, result: jb.Experiment) -> None:
        """Visualize the metrics of a benchmark result.

         Attributes:
             result (jb.Experiment): a benchmark result.
             parallelplot_axes_list (list[str]): a list of the names of all the metrics that can be plotted by `parallelplot_experiment`.
                 None until `parallelplot_experiment` is called.

        Example:
             Below is the code to boxplot the constraint violations.
             Check the docstring of each method for details.

            ```python
            import jijbench as jb
            import jijzept as jz
            from jijzept.sampler.openjij.sa_cpu import JijSAParameters
            from jijbench.visualization.metrics.plot import MetricsPlot

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
            from jijbench.visualization.metrics.plot import MetricsPlot
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
            from jijbench.visualization.metrics.plot import MetricsPlot
            from jijbench.visualization.metrics.utils import construct_experiment_from_samplesets
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
        title_list = _create_fig_title_list(metrics, title)

        fig_ax_list = []
        for i, (_, data) in enumerate(metrics.items()):
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
            from jijbench.visualization.metrics.plot import MetricsPlot

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
            from jijbench.visualization.metrics.plot import MetricsPlot
            from jijbench.visualization.metrics.utils import construct_experiment_from_samplesets

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
            f=_get_violations_dict,
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
        color_midpoint: float | None = None,
        additional_axes: list[str] | None = None,
        additional_axes_created_by_function: dict[str, Callable] | None = None,
        display_axes_list: list[str] | None = None,
        rename_map: dict[str, str] | None = None,
        axis_label_pos: AXIS_LABEL_POS | None = None,
        axis_label_fontsize: Number | None = None,
        title: str | None = None,
        height: Number | None = None,
        width: Number | None = None,
    ) -> plotly.graph_objects.Figure:
        """Plot the parallel plot of the experiment.

        Args:
            color_column_name (str | None): the column name, and the values from this column are used to assign color to mark.
                Defaults to samplemean_total_violations or objective if those columns exist.
            color_midpoint (float | None): the midpoint of the color scale. Defaults to the mean of the color column value.
            additional_axes (list[str] | None): A list of column names for additional axes.
                The conditions for available column names are that they are elements of self.result.table.columns and that the values from the column is number.
                Defaults to None.
            additional_axes_created_by_function (dict[str, Callable]): A list of dict, where the key is the label of the axis and the value is the callable.
                The callable is applied to `self.result.table` as self.result.table.apply(callable, axis=1), and the result is added to axes.
                The callable takes a `pd.Series` and returns a number.
                Defaults to None.
            display_axes_list (list[str] | None): A list of labels for the displayed axes. This argument allows you to select and sort the axes to display.
                Check the `parallelplot_axes_list` attribute for available axes. Defaults to all axes.
            rename_map (dict[str, str] | None): A dictionary where the key is the original axis label and the value is the user-specified axis label.
                Check the original axis labels in the `parallelplot_axes_list` attribute.
                Defaults is None, the original axis labels is displayed.
            axis_label_pos (AXIS_LABEL_POS | None): the position of the axis label. Only "top" or "bottom" are accepted. Defaults to top.
            axis_label_fontsize (Number | None): the fontsize of the axis label. Defaults to None.
            title (str | None): the title of the plot. Defaults to None.
            height (Number | None): the height of the plot. Defaults to None.
            width (Number | None): the width of the plot. Defaults to None.

        Returns:
            plotly.graph_objects.Figure: the parallel plot of the experiment.

        Examples:
            The following example is the most basic usage. A parallel plot of the benchmark results is displayed.

            ```python
            from itertools import product
            import jijbench as jb
            from jijbench.visualization.metrics.plot import MetricsPlot
            import jijzept as jz

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]

            onehot_time_multipliers = [0.01, 0.1]
            onehot_location_multipliers = [0.01, 0.1]
            multipliers = [
                {"onehot_time": onehot_time_multiplier,
                "onehot_location": onehot_location_multiplier}
                for onehot_time_multiplier, onehot_location_multiplier in product(onehot_time_multipliers, onehot_location_multipliers)
            ]
            config_path = XX
            sa_sampler = jz.JijSASampler(config=config_path)
            bench = jb.Benchmark(
                params = {
                    "model": [problem],
                    "feed_dict": [instance_data],
                    "multipliers": multipliers,
                },
                solver = [sa_sampler.sample_model],
            )
            result = bench()

            mp = MetricsPlot(result)
            fig = mp.parallelplot_experiment()
            ```

            You can change the appearance of the graph by performing the following operations on the `plotly.graph_objects.Figure` instance returned by `parallelplot_experiment`.
            The example below changes the fontsize of the range.
            For other operations, refer to the plotly reference, https://plotly.com/python/reference/parcoords/.

            ```python
            fig.update_traces(rangefont_size=15, selector=dict(type='parcoords'))
            fig.show()
            ```

            This example gives some arguments to `parallelplot_experiment`.
            `additional_axes` argument adds the execution_time column to the plot. This is the column of result.table and its element are number.
            `additional_axes_created_by_function` argument add the values calculated from result.table to the plot.
            `rename_map` insert line breaks for long data labels to make the charts easier to read. Line breaks are done with <br>.
            `axis_label_pos` is set to bottom to avoid the data label overlapping the figure due to line breaks.

            ```python
            from itertools import product
            import numpy as np
            import pandas as pd

            import jijbench as jb
            from jijbench.visualization.metrics.plot import MetricsPlot
            import jijzept as jz
            from jijzept.sampler.openjij.sa_cpu import JijSAParameters

            problem = jb.get_problem("TSP")
            instance_data = jb.get_instance_data("TSP")[0][1]

            onehot_time_multipliers = [0.01, 0.1]
            onehot_location_multipliers = [0.01, 0.1]
            multipliers = [
                {"onehot_time": onehot_time_multiplier,
                "onehot_location": onehot_location_multiplier}
                for onehot_time_multiplier, onehot_location_multiplier in product(onehot_time_multipliers, onehot_location_multipliers)
            ]
            config_path = "XX"
            sa_parameter = JijSAParameters(num_reads=30)
            sa_sampler = jz.JijSASampler(config=config_path)
            bench = jb.Benchmark(
                params = {
                    "model": [problem],
                    "feed_dict": [instance_data],
                    "parameters": [sa_parameter],
                    "multipliers": multipliers,
                },
                solver = [sa_sampler.sample_model],
            )
            result = bench()


            def get_num_reads_from_parameters(x: pd.Series) -> float:
                return x["parameters"].num_reads

            def calc_samplemean_energy(x: pd.Series) -> float:
                num_occ	= x["num_occurrences"]
                array = x["energy"]
                mean = np.sum(num_occ * array) / np.sum(num_occ)
                return mean

            mp = MetricsPlot(result)
            fig = mp.parallelplot_experiment(
                additional_axes=["execution_time"],
                additional_axes_created_by_function={
                    "num_reads": get_num_reads_from_parameters,
                    "samplemean_energy": calc_samplemean_energy,
                },
                rename_map={
                    "onehot_time_multiplier": "onehot_time<br>multiplier",
                    "onehot_location_multiplier": "onehot_location<br>multiplier",
                    "samplemean_objective": "samplemean<br>objective",
                    "samplemean_onehot_time_violations": "samplemean<br>onehot_time<br>violations",
                    "samplemean_onehot_location_violations": "samplemean<br>onehot_location<br>violations",
                    "samplemean_total_violations": "samplemean<br>total_violations",
                },
                axis_label_pos="bottom",
            )

            ```

        """
        if additional_axes is None:
            additional_axes = []
        if additional_axes_created_by_function is None:
            additional_axes_created_by_function = {}
        if axis_label_pos is None:
            axis_label_pos = "top"
        if not (axis_label_pos in ["top", "bottom"]):
            raise ValueError(
                f"axis_label_pos must be 'top' or 'bottom', but {axis_label_pos} is given."
            )

        result_table = self.result.table

        # The key is a column name (str), and the value is the data of each column (pd.Series).
        data_to_create_df_parallelplot = {}

        # Displayed data specified by user
        for display_column in additional_axes:
            # Check if the column exists and the elements is number.
            if not _df_has_number_column_target_name(result_table, display_column):
                raise TypeError(
                    f"{display_column}is not a column with number elements."
                )
            data_to_create_df_parallelplot[display_column] = result_table[
                display_column
            ]

        # Data generated by user custom functions
        for column_name, func in additional_axes_created_by_function.items():
            # TODO: デコレータで書き直した方が可読性が上がると思われる。エラーレイズと返り値がnumberであることのチェックをデコレータ内で行う
            try:
                data_to_create_df_parallelplot[column_name] = result_table.apply(
                    func, axis=1
                )
            except Exception as e:
                msg = f'An error occurred inside your function. Please check "{func.__name__}" in additional_axes_created_by_function. -> {e}'
                raise UserFunctionFailedError(msg)
            for value in data_to_create_df_parallelplot[column_name].values:
                if not isinstance(value, Number):
                    raise TypeError(
                        f"{column_name} is not a column with number elements."
                    )

        # multiplires (If self.result has a valid multipliers column)
        if _df_has_valid_multipliers_column(result_table):
            for constraint_name in result_table["multipliers"].values[0].keys():
                data_to_create_df_parallelplot[
                    constraint_name + "_multiplier"
                ] = result_table.apply(
                    _get_multiplier, axis=1, constraint_name=constraint_name
                )

        # num_feasible
        if _df_has_number_column_target_name(result_table, "num_feasible"):
            data_to_create_df_parallelplot["num_feasible"] = result_table[
                "num_feasible"
            ]

        # objective
        if _df_has_number_array_column_target_name(result_table, "objective"):
            data_to_create_df_parallelplot["samplemean_objective"] = result_table.apply(
                _calc_mean_of_array, axis=1, column_name="objective"
            )

        # violations
        for violation_column_name in result_table.columns[
            result_table.columns.str.contains("violations")
        ]:
            if _df_has_number_array_column_target_name(
                result_table, violation_column_name
            ):
                data_to_create_df_parallelplot[
                    "samplemean_" + violation_column_name
                ] = result_table.apply(
                    _calc_mean_of_array,
                    axis=1,
                    column_name=violation_column_name,
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

        if display_axes_list is None:
            display_axes_list = self.parallelplot_axes_list
        self.df_parallelplot_displayed = df_parallelplot_displayed = df_parallelplot[
            display_axes_list
        ]

        if color_column_name is None:
            if "samplemean_total_violations" in df_parallelplot_displayed.columns:
                color_column_name = "samplemean_total_violations"
            elif "samplemean_objective" in df_parallelplot_displayed.columns:
                color_column_name = "samplemean_objective"

        if color_midpoint is None and color_column_name is not None:
            color_midpoint = df_parallelplot_displayed[color_column_name].mean()

        fig = px.parallel_coordinates(
            df_parallelplot_displayed.reset_index(drop=True),
            color=color_column_name,
            labels=rename_map,
            color_continuous_scale=px.colors.diverging.Tealrose,
            color_continuous_midpoint=color_midpoint,
            title=title,
            height=height,
            width=width,
        )
        fig.update_traces(labelside=axis_label_pos, selector=dict(type="parcoords"))
        fig.update_traces(
            labelfont_size=axis_label_fontsize, selector=dict(type="parcoords")
        )
        fig.show()

        return fig

    @property
    def parallelplot_axes_list(self) -> list[str]:
        if hasattr(self, "df_parallelplot"):
            return list(self.df_parallelplot.columns)
        else:
            return []
