from __future__ import annotations

from typing import Union

from jijbench.components.dir import ExperimentResultDefaultDir

__all__ = []

class JijBenchObject:
    @classmethod
    def load(
        cls,
        *,
        benchmark_id: Union[int, str],
        experiment_id: Union[int, str] = None,
        autosave: bool = True,
        save_dir: str = ExperimentResultDefaultDir,
    ):
        pass

    @classmethod
    def load_dtypes(
        cls,
        *,
        benchmark_id: Union[int, str],
        experiment_id: Union[int, str] = None,
        autosave: bool = True,
        save_dir: str = ExperimentResultDefaultDir,
    ):
        pass
