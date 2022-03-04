import os
import json
import uuid
import dimod
import pickle
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from jijbench.experiment.artifact_parser import get_dimod_sampleset_items, get_jm_problem_decodedsamples_items


class Experiment:
    def __init__(
        self,
        experiment_id: Union[int, str] = None,
        benchmark_id: Union[int, str] = None,
        autosave: bool = True,
        autosave_dir: str = ".",
    ):
        self.autosave = autosave
        self.autosave_dir = autosave_dir

        self._table = _Table(experiment_id=experiment_id, benchmark_id=benchmark_id)
        self._artifact = {}
        self._dirs = _Dir(
            experiment_id=experiment_id,
            benchmark_id=benchmark_id,
            autosave=autosave,
            autosave_dir=autosave_dir,
        )

        # initialize table index
        self._table.current_index = 0

    @property
    def run_id(self):
        return self._table.run_id

    @property
    def experiment_id(self):
        return self._table.experiment_id

    @property
    def benchmark_id(self):
        return self._table.benchmark_id

    @property
    def table(self):
        return self._table.data

    @property
    def artifact(self):
        return self._artifact

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.close()
        pass

    def __next__(self):
        self.close()
        return self.run_id

    def start(self):
        self._table.run_id = uuid.uuid4()
        self._dirs.make_dirs(
            experiment_id=self.experiment_id, benchmark_id=self.benchmark_id
        )
        return self
    
    def close(self):
        self._table.current_index += 1
        self._table.run_id = uuid.uuid4()

    def stop(self):
        pass

    def store(
            self,
            results: Dict[str, Any],
            table_keys: Optional[List[str]]=None,
            artifact_keys: Optional[List[str]]=None,
            next_run: bool=True):
        """store results

        Args:
            results (Dict[str, Any]): ex. {"num_reads": 10, "results": sampleset}
            table_keys (list[str], optional): _description_. Defaults to None.
            artifact_keys (list[str], optional): _description_. Defaults to None.
            next_run (bool, optional): _description_. Defaults to True.
        """
        
        if table_keys is None:
            self.store_as_table(results)
        else:
            record = {k: results[k] for k in table_keys if k in results.keys()}
            self.store_as_table(record)

        if artifact_keys is None:
            self.store_as_artifact(results)
        else:
            artifact = {k: results[k] for k in artifact_keys if k in results.keys()}
            self.store_as_artifact(artifact)


    def store_as_table(self, record):
        index = self._table.current_index
        ids = self._table.get_id_columns()
        self._table.data.loc[index, ids] = [
            self.run_id,
            self.experiment_id,
            self.benchmark_id,
        ]
        record = self._reconstruct_record(record)
        for k, v in record.items():
            if isinstance(v, dict):
                v = json.dumps(v)
            elif isinstance(v, list):
                v = str(v)

            self._table.data.loc[index, k] = v
            self._table.data[k] = self._table.data[k].astype(type(v))

    def _reconstruct_record(self, record):
        new_record = {}
        for k, v in record.items():
            if isinstance(v, dimod.SampleSet):
                columns, values = get_dimod_sampleset_items(v)
                for new_k, new_v in zip(columns, values):
                    new_record[new_k] = new_v
            elif v.__class__.__name__ == "DecodedSamples":
                columns, values = get_jm_problem_decodedsamples_items(v)
                for new_k, new_v in zip(columns, values):
                    new_record[new_k] = new_v
            else:
                new_record[k] = v
        return new_record

    def store_as_artifact(self, artifact):
        self._artifact.update({self.run_id: artifact})

    def load(self, load_file=None):
        self._table.data = pd.read_csv(
            f"{self._dirs.table_dir}/table.csv",
            index_col=0,
        )
        artifact = {}
        dir_names = os.listdir(self._dirs.artifact_dir)
        for d in dir_names:
            load_dir = f"{self._dirs.artifact_dir}/{d}"
            if os.path.isdir(load_dir):
                with open(f"{load_dir}/artifact.pkl", "rb") as f:
                    artifact[d] = pickle.load(f)
        self._artifact = artifact

    def load_table(self, load_file):
        self._table.data = pd.read_csv(load_file, index_col=0)

    def load_artifact(self, load_file):
        with open(load_file, "rb") as f:
            self._artifact = pickle.load(f)

    def save(self, save_file=None):
        self._table.data.to_csv(f"{self._dirs.table_dir}/table.csv")
        for run_id, v in self._artifact.items():
            save_dir = f"{self._dirs.artifact_dir}/{run_id}"
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/artifact.pkl", "wb") as f:
                pickle.dump(v, f)

    def save_table(self, save_file):
        self._table.data.to_csv(save_file)

    def save_artifact(self, save_file):
        with open(save_file, "wb") as f:
            pickle.dump(self._artifact, f)


class _Table:
    id_dtypes = {
        "run_id": object,
        "experiment_id": object,
        "benchmark_id": object,
    }
    energy_dtypes = {
        "energy": object,
        "energy_min": float,
        "energy_mean": float,
        "energy_std": float,
    }

    objective_dtypes = {
        "objective": object,
        "obj_min": float,
        "obj_mean": float,
        "obj_std": float,
    }

    num_dtypes = {
        "num_occurances": object,
        "num_feasible": int,
        "num_samples": int,
    }

    violation_dtypes = {
        "{const_name}_violations": object,
        "{const_name}_violation_min": float,
        "{const_name}_violation_mean": float,
        "{const_name}_violation_std": float,
    }

    __dtypes_names = [
        "id_dtypes",
        "energy_dtypes",
        "objective_dtypes",
        "num_dtypes",
        "violation_dtypes",
    ]

    def __init__(self, run_id=None, experiment_id=None, benchmark_id=None):
        columns = self.get_columns()
        self._data = pd.DataFrame(columns=columns)
        self._current_index = 0

        self.run_id = run_id
        self.experiment_id = experiment_id
        self.benchmark_id = benchmark_id

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def current_index(self):
        return self._current_index

    @current_index.setter
    def current_index(self, index):
        self._current_index = index

    def get_columns(self):
        c = []
        for name in self.__dtypes_names:
            c += list(getattr(self, name).keys())
        return c

    def get_id_columns(self):
        return list(self.id_dtypes.keys())

    def get_energy_columns(self):
        return list(self.energy_dtypes.keys())

    def get_objective_columns(self):
        return list(self.objective_dtypes.keys())

    def get_num_columns(self):
        return list(self.num_dtypes.keys())

    def get_violation_columns(self):
        return list(self.violation_dtypes.keys())

    def rename_violation_columns(self, const_name):
        columns = self.get_violation_columns()
        for i, c in enumerate(columns):
            columns[i] = c.format(const_name=const_name)
        return columns

    def get_dtypes(self):
        t = {}
        for name in self.__dtypes_names:
            t |= getattr(self, name)
        return t


class _Dir:
    _dir_template = "{autosave_dir}/benchmark_{benchmark_id}/{kind}/{experiment_id}"

    def __init__(self, experiment_id, benchmark_id, autosave, autosave_dir):
        self.experiment_id = experiment_id
        self.benchmark_id = benchmark_id
        self.autosave = autosave
        self.autosave_dir = autosave_dir

        self._table_dir = self._dir_template.format(
            autosave_dir=self.autosave_dir,
            benchmark_id=self.benchmark_id,
            kind="tables",
            experiment_id=self.experiment_id,
        )
        self._artifact_dir = self._dir_template.format(
            autosave_dir=self.autosave_dir,
            benchmark_id=self.benchmark_id,
            kind="artifact",
            experiment_id=self.experiment_id,
        )

    @property
    def table_dir(self):
        return self._table_dir

    @property
    def artifact_dir(self):
        return self._artifact_dir

    def make_dirs(self, experiment_id=None, benchmark_id=None):
        if self.autosave:
            self._table_dir = self._rename_dir(
                kind="tables", experiment_id=experiment_id, benchmark_id=benchmark_id
            )
            self._artifact_dir = self._rename_dir(
                kind="artifact", experiment_id=experiment_id, benchmark_id=benchmark_id
            )
            os.makedirs(self._table_dir, exist_ok=True)
            os.makedirs(self._artifact_dir, exist_ok=True)

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

        d = self._dir_template.format(
            autosave_dir=self.autosave_dir,
            benchmark_id=benchmark_id,
            kind=kind,
            experiment_id=experiment_id,
        )
        return d


if __name__ == "__main__":
    """# Example 1
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
        print(experiment.table)"""

    # Example 3
    import jijzept as jz
    import jijmodeling as jm

    d = jm.Placeholder("d", dim=1)
    n = d.shape[0]

    x = jm.Binary("x", shape=(n,))

    problem = jm.Problem("sample")

    i = jm.Element("i", (0, n))

    problem += jm.Sum(i, d[i] * x[i])
    problem += jm.Constraint("onehot", x[:] == 1)
    problem += jm.Constraint("onehot2", x[:] == 1)

    data = {"d": [1, 2, -1, -3, -1]}

    sampler = jz.JijSASampler(config="/home/azureuser/.config/jijzept/config.toml")
    response = sampler.sample_model(problem, data, {"onehot": 4}, num_reads=5)
    decoded = problem.decode(response, data, {})
    sample_artifacts = {"hoge": {"fuga": 1}}

    experiment = Experiment(benchmark_id="test")
    with experiment.start() as e:
        for param in [10, 100, 1000]:
            for step in range(3):
                # solverは上のsample_responseを返す想定
                # sample_response = solver()
                # experiment.tableに登録するrecordを辞書型で作成
                record = {
                    "step": step,
                    "param": param,
                    # "results": response,
                    "results": decoded,
                }
                experiment.store(
                    record,
                    table_keys=["param", "reuslts"],
                    artifact_keys=["step", "results"],
                )
    experiment.save()

    experiment_id = experiment.experiment_id
    experiment = Experiment(experiment_id=experiment_id, benchmark_id="test")
    experiment.load()

    print(experiment.table)
    print()
    print(experiment.artifact)
