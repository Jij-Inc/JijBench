from pkgutil import extend_path

__path__ = extend_path(__path__, __name__)

import jijbench.exceptions.exceptions as exceptions

from jijbench.exceptions.exceptions import (
    ConcurrentFailedError,
    SolverFailedError,
)

__all__ = [
    "exceptions",
    "ConcurrentFailedError",
    "SolverFailedError",
]
