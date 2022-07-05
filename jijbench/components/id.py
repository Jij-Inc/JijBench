from __future__ import annotations

import uuid

__all__ = []


class ID:
    """ID template"""

    def __init__(self, *, benchmark_id=None, experiment_id=None):
        if experiment_id is None:
            experiment_id = str(uuid.uuid4())
        if benchmark_id is None:
            benchmark_id = str(uuid.uuid4())
        self._run_id = None
        self._experiment_id = experiment_id
        self._benchmark_id = benchmark_id

    @property
    def run_id(self):
        return self._run_id

    @run_id.setter
    def run_id(self, run_id):
        self._run_id = run_id

    @property
    def experiment_id(self):
        return self._experiment_id

    @experiment_id.setter
    def experiment_id(self, experiment_id):
        self._experiment_id = experiment_id

    @property
    def benchmark_id(self):
        return self._benchmark_id

    @benchmark_id.setter
    def benchmark_id(self, benchmark_id):
        self._benchmark_id = benchmark_id

    def reset(self, *, kind="experiment"):
        setattr(self, f"_{kind}_id", str(uuid.uuid4()))
