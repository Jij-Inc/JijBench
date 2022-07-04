from __future__ import annotations

import os

ExperimentResultDefaultDir = "./.jb_results"


class Dir:
    """Directory template"""

    _dir_template = "{save_dir}/benchmark_{benchmark_id}/{experiment_id}/{kind}"

    def __init__(self, *, benchmark_id, experiment_id, autosave, save_dir):
        self.experiment_id = experiment_id
        self.benchmark_id = benchmark_id
        self.autosave = autosave
        self.save_dir = save_dir

        self._benchmark_dir = f"{self.save_dir}/benchmark_{self.benchmark_id}"
        self._experiment_dir = f"{self._benchmark_dir}/{self.experiment_id}"

        self._table_dir: str = self._dir_template.format(
            save_dir=self.save_dir,
            benchmark_id=self.benchmark_id,
            kind="tables",
            experiment_id=self.experiment_id,
        )
        self._artifact_dir: str = self._dir_template.format(
            save_dir=self.save_dir,
            benchmark_id=self.benchmark_id,
            kind="artifact",
            experiment_id=self.experiment_id,
        )

    @property
    def benchmark_dir(self) -> str:
        return self._benchmark_dir

    @property
    def experiment_dir(self) -> str:
        return self._experiment_dir

    @property
    def table_dir(self) -> str:
        return self._table_dir

    @property
    def artifact_dir(self) -> str:
        return self._artifact_dir

    def make_dirs(self, run_id, experiment_id=None, benchmark_id=None):
        self._table_dir = self._rename_dir(
            kind="tables", experiment_id=experiment_id, benchmark_id=benchmark_id
        )
        self._artifact_dir = self._rename_dir(
            kind="artifact", experiment_id=experiment_id, benchmark_id=benchmark_id
        )
        os.makedirs(self._table_dir, exist_ok=True)
        os.makedirs(f"{self._artifact_dir}/{run_id}", exist_ok=True)

    def _rename_dir(
        self,
        kind,
        experiment_id=None,
        benchmark_id=None,
    ):
        if experiment_id is None:
            experiment_id = self.experiment_id
        if benchmark_id is None:
            benchmark_id = self.benchmark_id

        self._benchmark_dir = f"{self.save_dir}/benchmark_{benchmark_id}"
        self._experiment_dir = f"{self._benchmark_dir}/{experiment_id}"

        d = self._dir_template.format(
            save_dir=self.save_dir,
            benchmark_id=benchmark_id,
            kind=kind,
            experiment_id=experiment_id,
        )
        return d
