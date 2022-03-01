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
        self._table = pd.DataFrame(columns=self._id_names)
        self._table_dtypes = {"run_id": int, "experiment_id": type(self.experiment_id)}

        if autosave:
            os.makedirs(autosave_dir, exist_ok=True)
            if benchmark_id is None:
                benchmark_id = sum(
                    [
                        os.path.isdir(f"{self.autosave_dir}/{d}")
                        for d in os.listdir(self.autosave_dir)
                        if "benchmark" in d
                    ]
                )
            self.benchmark_id = benchmark_id
            self.table_dir = f"{self.autosave_dir}/benchmark_{self.benchmark_id}/tables"
        else:
            self.benchmark_id = None
            self.table_dir = None

    @property
    def table(self):
        return self._table

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        pass

    def __next__(self):
        self.run_id += 1
        return self.run_id

    def insert_into_table(self, record, replace=True):
        self._table.loc[self.run_id, self._id_names] = [self.run_id, self.experiment_id]
        for k, v in record.items():
            if isinstance(v, dict):
                v = json.dumps(v)
            if isinstance(v, list):
                v = str(v)

            self._table.loc[self.run_id, k] = v
            self._table[k] = self._table[k].astype(type(v))
            self._table_dtypes[k] = type(v)
        if replace:
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
    # Example 1
    # 最も単純な使い方
    # ユーザ定義のsolverの帰り値（何でも良い）
    sample_response = {"hoge": {"fuga": 1}}

    with Experiment() as experiment:
        for param in [10, 100, 1000]:
            for step in range(3):
                # solverは上のsample_responseを返す想定
                # sample_response = solver()
                # experiment.tableに登録するrecordを辞書型で作成
                record = {
                    "step": step,
                    "param": param,
                    "results": sample_response,
                }
                experiment.insert_into_table(record)
        experiment.save()
        print(experiment.table)
    print()

    # Example 2
    # 実験結果を保存したい場所を指定する。
    # experiment_idとbenchmark_idを明示的に指定し保存した結果の読み込みを分かりやすくする。
    save_dir = "/home/azureuser/data/jijbench"
    experiment_id = "test"
    benchmark_id = 0
    
    # 実験したいパラメータ（solverに渡すパラメータ）
    params = [10, 100, 1000]
    steps = range(3)

    with Experiment(
        experiment_id=experiment_id, benchmark_id=benchmark_id, autosave_dir=save_dir
    ) as experiment:
        for param in params:
            for step in steps:
                # sample_response = solver()
                record = {
                    "step": step,
                    "param": param,
                    "results": sample_response,
                }
                experiment.insert_into_table(record)
        experiment.save()

    # 以前実験した結果をloadする。experiment_idとbenchmark_idを覚えていればいつでも読み込みできる。
    # もちろんファイル名を直接指定しても良い。その場合はautosave=Falseにしてloadでファイル名を指定する。
    with Experiment(
        experiment_id=experiment_id, benchmark_id=benchmark_id, autosave_dir=save_dir
    ) as experiment:
        experiment.load()
        print(experiment.table)
