from abc import ABCMeta, abstractmethod


class Figure(metaclass=ABCMeta):
    @abstractmethod
    def show(self):
        raise NotImplementedError
