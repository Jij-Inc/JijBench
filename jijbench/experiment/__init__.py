from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

import jijbench.experiment._parser

from jijbench.experiment.experiment import Experiment

__all__ = []
