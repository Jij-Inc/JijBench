import pytest

import jijbench as jb
from jijbench.components import ExperimentResultDefaultDir, Table
from jijbench.exceptions import LoadFailedError


def test_load_dtypes_invalid_id():
    INVALID_BENCHMARK_ID = "invalid_benchmark_id"
    INVALID_EXPERIMENT_ID = "invalid_experiment_id"

    def func1(x):
        return 2 * x

    bench = jb.Benchmark(params={"x": [1, 2, 3]}, solver=func1, benchmark_id="test")
    bench.run()

    experiment_id = bench.table["experiment_id"].values[0]

    with pytest.raises(LoadFailedError):
        dtypes = Table.load_dtypes(
            benchmark_id=INVALID_BENCHMARK_ID,
            experiment_id=INVALID_EXPERIMENT_ID,
            autosave=True,
            save_dir=ExperimentResultDefaultDir,
        )
