from __future__ import annotations


class JijBenchmarkError(Exception):
    """
    Exception for JijBenchmark Errors related to JijBenchmark inherit from this.
    Exception class.
    """


class ConcurrentFailedError(JijBenchmarkError):
    pass


class SolverFailedError(JijBenchmarkError):
    pass


class LoadFailedError(JijBenchmarkError):
    pass


class StoreResultFailedError(JijBenchmarkError):
    pass
