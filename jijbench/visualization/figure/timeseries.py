from __future__ import annotations

from collections import OrderedDict
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from jijbench.visualization.figure.interface import Figure


class TimeSeries(Figure):
    """Visualize time series.

    Attributes:
        data (OrderedDict):the dict of time series. the key is label, and the value is tuple of x and y.
        fig_ax (tuple[matplotlib.figure.Figure, matplotlib.axes.Subplot]): Figure and Axes of matplotlib. Available after show method is called.
    Example:
        The code below plots a linear function and a quadratic function.
        The style of the graph (e.g. color) can be changed by arguments of the show method.

        ```python
        >>> import numpy as np
        >>> from jijbench.figure.timeseries import TimeSeries

        >>> x1 = np.arange(-10, 11, 1)
        >>> y1 = x1 + 1

        >>> x2 = np.arange(-10, 11, 1)
        >>> y2 = x2 ** 2

        >>> timeseries = TimeSeries()
        >>> timeseries.add_data("linear", x1, y1)
        >>> timeseries.add_data("quadratic", x2, y2)
        >>> timeseries.show(color_list=["red", "green"])
        ```
    """

    def __init__(
        self,
    ) -> None:
        self.data = OrderedDict()
        self._fig_ax = None

    def add_data(
        self,
        label: str,
        plot_x: list[int | float] | npt.NDArray,
        plot_y: list[int | float] | npt.NDArray,
    ) -> None:
        """Add time series data to data attribute for plot.

        Args:
            label (str): the label of the time series.
            plot_x (list[int | float] | npt.NDArray): the 1D list of horizontal axis value (the list of time).
            plot_y (list[int | float] | npt.NDArray): the 1D list of vertical axis value.
        """
        if isinstance(plot_x, np.ndarray):
            plot_x = plot_x.tolist()
        if isinstance(plot_y, np.ndarray):
            plot_y = plot_y.tolist()

        if len(plot_x) != len(plot_y):
            raise ValueError("plot_x and plot_y must be the same length.")
        self.data.update([(label, (plot_x, plot_y))])

    def show(
        self,
        figsize: tuple[int | float] | None = None,
        title: str | None = None,
        color_list: list | None = None,
        alpha_list: list[float] | None = None,
        linestyle_list: list[str] | None = None,
        marker_list: list[str] | None = None,
        xlabel: str | None = None,
        ylabel: str | None = None,
        xticks: list[int | float] | None = None,
        yticks: list[int | float] | None = None,
    ):
        """Plot time series data which you passed to the add_data method.

        The arguments of the show method are passed to the plot of matplotlib.

        Args:
            figsize (tuple[int | float] | None): the size of figure. The default uses matplotlib's default value.
            title (str | None): the title of figure. Defaults to "time series".
            color_list (list | None): the list of plot line color. The default uses matplotlib's default value.
            alpha_list (list[float] | None): the list of plot line transparency. The default is 1.0 for each plot line.
            linestyle_list (list[str] | None): the list of plot line linestyle. The default is "solid" for each plot line.
            marker_list (list[str] | None): the list of plot line marker. The default is "o" for each plot line.
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
            title = "time series"

        if (color_list is not None) and (len(color_list) != len(data)):
            raise ValueError("color_list and data must be same length.")

        if alpha_list is None:
            alpha_list = [1.0] * len(data)
        elif len(alpha_list) != len(data):
            raise ValueError("alpha_list and data must be same length.")

        if linestyle_list is None:
            linestyle_list = ["solid"] * len(data)
        elif len(linestyle_list) != len(data):
            raise ValueError("linestyle_list and data must be same length.")

        if marker_list is None:
            marker_list = ["o"] * len(data)
        elif len(marker_list) != len(data):
            raise ValueError("marker_list and data must be same length.")

        if xlabel is None:
            xlabel = "time"

        fig, ax = plt.subplots(figsize=figsize)
        fig.suptitle(title)

        for i, (label, plot_data) in enumerate(data.items()):
            x, y = plot_data
            color = None if color_list is None else color_list[i]
            ax.plot(
                x,
                y,
                label=label,
                color=color,
                alpha=alpha_list[i],
                linestyle=linestyle_list[i],
                marker=marker_list[i],
            )

        ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if xticks is not None:
            ax.set_xticks(xticks)
        if yticks is not None:
            ax.set_yticks(yticks)
        ax.legend()

        self._fig_ax = (fig, ax)

    @property
    def fig_ax(self):
        if self._fig_ax is None:
            raise AttributeError(
                "fig_ax attribute is available after show method is called."
            )
        else:
            return self._fig_ax
