from __future__ import annotations


class JijBenchmarkError(Exception):
    """
    Exception for JijBenchmark Errors related to JijBenchmark inherit from this.
    Exception class.
    """


class JijBenchmarkUnsupportedProblemError(JijBenchmarkError):
    pass


class JijBenchmarkUnsupportedInstanceDataError(JijBenchmarkError):
    pass
