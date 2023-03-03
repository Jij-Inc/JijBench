from abc import ABCMeta, abstractmethod


class Figure(metaclass=ABCMeta):
    # Display a figure using matplotlib
    @abstractmethod
    def show(self):
        raise NotImplementedError

    # Returns the matplotlib figure and axes created in show method.
    @property
    @abstractmethod
    def fig_ax(self):
        raise NotImplementedError
