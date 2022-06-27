from typing import Union
from .dir import ExperimentResultDefaultDir


class JijBenchObject:
    @classmethod
    def load(
        cls,
        *,
        benchmark_id: Union[int, str],
        experiment_id: Union[int, str] = None,
        autosave: bool = True,
        save_dir: str = ExperimentResultDefaultDir
    ):
        pass

    @classmethod
    def load_dtypes(
        cls,
        *,
        benchmark_id: Union[int, str],
        experiment_id: Union[int, str] = None,
        autosave: bool = True,
        save_dir: str = ExperimentResultDefaultDir
    ):
        pass
