import os
import json
from typing import Union
import pandas as pd


class Experiment:
    def __init__(
        self,
        run_id: int = 0,
        experiment_id: Union[int, str] = 0,
        benchmark_id: Union[int, str] = None,
        autosave: bool = True,
        autosave_dir: str = ".",
    ):
        self.run_id = run_id
        self.experiment_id = experiment_id
        self.autosave = autosave
        self.autosave_dir = autosave_dir

        self._id_names = ["run_id", "experiment_id"]
        self._table = None
        self._table_dtypes = {"run_id": int, "experiment_id": type(self.experiment_id)}

        if autosave:
            os.makedirs(autosave_dir, exist_ok=True)
            if benchmark_id is None:
                benchmark_id = len(os.listdir(self.autosave_dir))
            self.benchmark_id = benchmark_id
            self.table_dir = f"{self.autosave_dir}/benchmark_{self.benchmark_id}/tables"
        else:
            self.benchmark_id = None
            self.table_dir = None

    @property
    def table(self):
        return self._table

    def __enter__(self, *args, **kwargs):
        self._table = pd.DataFrame(columns=self._id_names)
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def __next__(self):
        self.run_id += 1
        return self.run_id

    def insert_into_table(self, record):
        self._table.loc[self.run_id, self._id_names] = [self.run_id, self.experiment_id]
        for k, v in record.items():
            if isinstance(v, dict):
                v = json.dumps(v)

            self._table.loc[self.run_id, k] = v
            self._table[k] = self._table[k].astype(type(v))
            self._table_dtypes[k] = type(v)
        next(self)

    def load(self, load_file=None):
        if self.autosave:
            self._table = pd.read_csv(
                f"{self.table_dir}/experiment_id_{self.experiment_id}.csv", index_col=0
            )
        else:
            self._table = pd.read_csv(load_file, index_col=0)

    def save(self, save_file=None):
        if self.autosave:
            os.makedirs(self.table_dir, exist_ok=True)
            self._table.to_csv(
                f"{self.table_dir}/experiment_id_{self.experiment_id}.csv"
            )
        else:
            self._table.to_csv(save_file)


if __name__ == "__main__":
    # ユーザ定義のsolverの帰り値（何でも良い）
    sample_response = {"hoge": {"fuga": 1}}

    # 実験したいパラメータ（solverに渡すパラメータ）
    params_1 = [10, 100, 1000]
    params_2 = [5, 10, 15]
    steps = range(3)

    # 実験結果を保存したい場所
    save_dir = "/home/azureuser/data/jijbench"
    experiment_id = "test"
    benchmark_id = 0
    with Experiment(
        experiment_id=experiment_id, benchmark_id=benchmark_id, autosave_dir=save_dir
    ) as experiment:
        for p1 in params_1:
            for p2 in params_2:
                for step in steps:
                    # solverは上のsample_responseを返す想定
                    # sample_response = solver()

                    # experiment.tableに登録するrecordを辞書型で作成
                    record = {
                        "step": step,
                        "param_1": p1,
                        "param_2": p2,
                        "results": sample_response,
                    }
                    experiment.insert_into_table(record)
        experiment.save()

    # 以前実験した結果をloadしたい場合
    with Experiment(
        experiment_id=experiment_id, benchmark_id=benchmark_id, autosave_dir=save_dir
    ) as experiment:
        experiment.load()
