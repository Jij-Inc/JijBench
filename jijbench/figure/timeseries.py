from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from typing import List, Optional, Tuple, Union

from jijbench.figure.interface import Figure


class TimeSeries(Figure):
    """Visualize time series.

    Attributes:
        data (OrderedDict):the dict of time series. the key is label, and the values is tuple of x and y.
        fig_ax (Tuple[matplotlib.figure.Figure, matplotlib.axes.Subplot]): Figure and Axes of matplotlib. Available after show method is called.
    Example:
        The code below plots a linear function and a quadratic function.
        The style of the graph (e.g. color) can be changed by arguments of the show method.

        '''python
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
        '''

    """

    def __init__(
        self,
    ) -> None:
        self.data = OrderedDict()
        self._fig_ax = None

    def add_data(
        self,
        label: str,
        plot_x: Union[List[Union[int, float]], npt.NDArray],
        plot_y: Union[List[Union[int, float]], npt.NDArray],
    ) -> None:
        """Add time series data to data attribute for plot.

        Args:
            label (str): the label of the time series.
            plot_x (Union[List[Union[int, float]], npt.NDArray]): the 1D list of horizontal axis value (the list of time).
            plot_y (Union[List[Union[int, float]], npt.NDArray]): the 1D list of vertical axis value.
        """
        plot_x = plot_x.tolist() if type(plot_x) == np.ndarray else plot_x
        plot_y = plot_y.tolist() if type(plot_y) == np.ndarray else plot_y

        if len(plot_x) != len(plot_y):
            raise ValueError("plot_x and plot_y must be the same length.")
        self.data.update([(label, (plot_x, plot_y))])

    def show(
        self,
        figsize: Optional[Tuple[Union[int, float]]] = None,
        title: Optional[str] = None,
        color_list: Optional[List] = None,
        alpha_list: Optional[List[float]] = None,
        linestyle_list: Optional[List[str]] = None,
        marker_list: Optional[List[str]] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        xticks: Optional[List[Union[int, float]]] = None,
        yticks: Optional[List[Union[int, float]]] = None,
    ):
        """Plot time series data which you passed to the add_data method.

        The arguments of the show method are passed to the plot of matplotlib.

        Args:
            figsize (Optional[Tuple[Union[int, float]]]): the size of figure. The default uses matplotlib's default value.
            title (Optional[str]): the title of figure. Defaults to "time series".
            color_list (Optional[List]): the list of plot line color. The default uses matplotlib's default value.
            alpha_list (Optional[List[float]]): the list of plot line transparency. The default is 1.0 for each plot line.
            linestyle_list (Optional[List[str]]): the list of plot line linestyle. The default is "solid" for each plot line.
            marker_list (Optional[List[str]]): the list of plot line marker. The default is "o" for each plot line.
            xlabel (Optional[str]): the xlabel of figure. Defaults to None.
            ylabel (Optional[str]): the ylabel of figure. Defaults to None.
            xticks (Optional[List[Union[int, float]]]): the xticks of figure. The default uses matplotlib's default.
            yticks (Optional[List[Union[int, float]]]): the yticks of figure. The default uses matplotlib's default.
        """
        data = self.data

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

        if xlabel is not None:
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
