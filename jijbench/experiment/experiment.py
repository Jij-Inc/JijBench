import datetime
import os
import uuid
import dimod
import json
import pickle
import re
import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Union
from jijbench.experiment.artifact_parser import (
    get_dimod_sampleset_items,
    get_jm_problem_decodedsamples_items,
)

ExperimentResultDefaultDir = "./.jb_results"
np.set_printoptions(threshold=np.inf)


class Experiment:
    """Experiment class
    Manage experimental results as Dataframe and artifact (python objects).
    """

    def __init__(
        self,
        experiment_id: Union[int, str] = None,
        benchmark_id: Union[int, str] = None,
        autosave: bool = True,
        save_dir: str = ExperimentResultDefaultDir,
    ):
        self.autosave = autosave
        self.save_dir = save_dir

        if benchmark_id is None:
            benchmark_id = uuid.uuid4()
        if experiment_id is None:
            experiment_id = uuid.uuid4()

        self._table = _Table(
            experiment_id=str(experiment_id), benchmark_id=str(benchmark_id)
        )
        self._artifact = {}
        self._artifact_timestamp = {}
        self._dirs = _Dir(
            experiment_id=str(experiment_id),
            benchmark_id=str(benchmark_id),
            autosave=autosave,
            save_dir=save_dir,
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

    def start(self):
        self._table.run_id = str(uuid.uuid4())
        self._dirs.make_dirs()
        return self

    def close(self):
        if self.autosave:
            record = self._table.data.loc[self._table.current_index].dropna().to_dict()
            self.add_record_to_csv(record)

            run_id = self.run_id
            if run_id in self._artifact.keys():
                save_dir = f"{self._dirs.artifact_dir}/{run_id}"
                os.makedirs(save_dir, exist_ok=True)
                with open(f"{save_dir}/artifact.pkl", "wb") as f:
                    pickle.dump(self._artifact[run_id], f)

                timestamp = self._artifact_timestamp[run_id]
                with open(f"{save_dir}/timestamp.txt", "w") as f:
                    f.write(str(timestamp))
        self.save_dtypes()
        self._table.current_index += 1

    def store(
        self,
        results: Dict[str, Any],
        table_keys: Optional[List[str]] = None,
        artifact_keys: Optional[List[str]] = None,
        timestamp: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
    ):
        """store results

        Args:
            results (Dict[str, Any]): ex. {"num_reads": 10, "results": sampleset}
            table_keys (list[str], optional): _description_. Defaults to None.
            artifact_keys (list[str], optional): _description_. Defaults to None.
            next_run (bool, optional): _description_. Defaults to True.
        """

        if timestamp is None:
            _timestamp = pd.Timestamp.now()
        else:
            _timestamp = pd.Timestamp(timestamp)

        if table_keys is None:
            self.store_as_table(results, timestamp=_timestamp)
        else:
            record = {k: results[k] for k in table_keys if k in results.keys()}
            self.store_as_table(record, timestamp=_timestamp)

        if artifact_keys is None:
            self.store_as_artifact(results, timestamp=_timestamp)
        else:
            artifact = {k: results[k] for k in artifact_keys if k in results.keys()}
            self.store_as_artifact(artifact, timestamp=_timestamp)

    def store_as_table(
        self,
        record: dict,
        timestamp: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
    ):
        """store as table

        Args:
            record (dict): record
            timestamp (pd.Timestamp | datetime.datetime, optional): time stamp. Defaults to None.
        """
        index = self._table.current_index
        ids = self._table.get_id_columns()
        if timestamp is None:
            _timestamp = pd.Timestamp.now()
        else:
            _timestamp = pd.Timestamp(timestamp)

        ids_data = [self.run_id, self.experiment_id, self.benchmark_id, _timestamp]
        self._table.data.loc[index, ids] = ids_data
        record = self._reconstruct_record(record)
        for key, value in record.items():
            value_type = type(value)
            if isinstance(value, dict):
                self._table.data.at[index, key] = object
                value_type = object
            elif isinstance(value, list):
                self._table.data.at[index, key] = object
                value_type = object
            elif isinstance(value, np.ndarray):
                self._table.data.at[index, key] = object
                value_type = object
            record[key] = value
            self._table.data.at[index, key] = value
            self._table.data[key] = self._table.data[key].astype(value_type)

    def _reconstruct_record(self, record):
        new_record = {}
        for k, v in record.items():
            if isinstance(v, dimod.SampleSet):
                columns, values = get_dimod_sampleset_items(self, v)
                for new_k, new_v in zip(columns, values):
                    new_record[new_k] = new_v
            elif v.__class__.__name__ == "DecodedSamples":
                columns, values = get_jm_problem_decodedsamples_items(self, v)
                for new_k, new_v in zip(columns, values):
                    new_record[new_k] = new_v
            else:
                new_record[k] = v
        return new_record

    def store_as_artifact(
        self,
        artifact,
        timestamp: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
    ):

        if timestamp is None:
            timestamp = pd.Timestamp.now()
        else:
            timestamp = pd.Timestamp(timestamp)

        self._artifact_timestamp.update({self.run_id: timestamp})
        self._artifact.update({self.run_id: artifact})

    @classmethod
    def load(
        cls,
        experiment_id: Union[int, str],
        benchmark_id: Union[int, str],
        autosave: bool = True,
        save_dir: str = ExperimentResultDefaultDir,
    ) -> "Experiment":

        experiment = Experiment(
            experiment_id=experiment_id,
            benchmark_id=benchmark_id,
            autosave=autosave,
            save_dir=save_dir,
        )
        experiment._table.data = pd.read_csv(
            f"{experiment._dirs.table_dir}/table.csv", index_col=0
        )
        dtypes = experiment.load_dtypes(f"{experiment._dirs.experiment_dir}/dtypes.pkl")
        experiment._table.as_finetype(dtypes)

        artifact = {}
        artifact_timestamp = {}
        dir_names = os.listdir(experiment._dirs.artifact_dir)
        for d in dir_names:
            load_dir = f"{experiment._dirs.artifact_dir}/{d}"
            if os.path.isdir(load_dir):
                with open(f"{load_dir}/artifact.pkl", "rb") as f:
                    artifact[d] = pickle.load(f)
                with open(f"{load_dir}/timestamp.txt", "r") as f:
                    artifact_timestamp[d] = pd.Timestamp(f.read())

        experiment._artifact = artifact
        experiment._artifact_timestamp = artifact_timestamp
        return experiment

    @classmethod
    def load_table(cls, load_file):
        NotImplementedError

    @classmethod
    def load_artifact(cls, load_file):
        NotImplementedError

    @classmethod
    def load_dtypes(self, load_file):
        with open(load_file, "rb") as f:
            return pickle.load(f)

    def save(self):
        self._table.data.to_csv(f"{self._dirs.table_dir}/table.csv")
        for run_id, v in self._artifact.items():
            save_dir = f"{self._dirs.artifact_dir}/{run_id}"
            os.makedirs(save_dir, exist_ok=True)
            with open(f"{save_dir}/artifact.pkl", "wb") as f:
                pickle.dump(v, f)

            timestamp = self._artifact_timestamp[run_id]
            with open(f"{save_dir}/timestamp.txt", "w") as f:
                f.write(str(timestamp))

    def save_table(self, save_file):
        self._table.data.to_csv(save_file)

    def save_artifact(self, save_file):
        with open(save_file, "wb") as f:
            pickle.dump(self._artifact, f)

    def save_dtypes(self):
        index = self._table.data.loc[self._table.current_index].index
        dtypes = {}
        for i, v in zip(index, self._table.data.loc[self._table.current_index]):
            dtypes[i] = type(v)
        with open(f"{self._dirs.experiment_dir}/dtypes.pkl", "wb") as f:
            pickle.dump(dtypes, f)

    def add_record_to_csv(self, record: dict):
        df = pd.DataFrame({key: [value] for key, value in record.items()})
        os.makedirs(self._dirs.table_dir, exist_ok=True)
        file_name = f"{self._dirs.table_dir}/table.csv"
        df.to_csv(file_name, mode="a", header=not os.path.exists(file_name))


class _Table:
    """Table template"""

    id_dtypes = {
        "run_id": object,
        "experiment_id": object,
        "benchmark_id": object,
        "timestamp": pd.Timestamp,
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

    _dtypes_names = [
        "id_dtypes",
        "energy_dtypes",
        "objective_dtypes",
        "num_dtypes",
        "violation_dtypes",
    ]

    def __init__(self, experiment_id, benchmark_id):
        columns = self.get_default_columns()
        self._data = pd.DataFrame(columns=columns)
        self._current_index = 0

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

    def get_default_columns(self):
        c = []
        for name in self._dtypes_names:
            if "violation_dtypes" not in name:
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
        for name in self._dtypes_names:
            t |= getattr(self, name)
        return t

    def as_finetype(self, dtypes):
        for col, dtype in dtypes.items():
            if dtype == list:
                self.to_list([col])
            elif dtype == np.ndarray:
                self.to_numpy([col])
            elif dtype == dict:
                self.to_dict([col])

    def to_numpy(self, labels=None):
        if labels is None:
            for col in self._data:
                self._data[col] = (
                    self._data[col]
                    .map(self._replace_numpystr)
                    .map(self._to_list)
                    .map(np.array)
                )
        else:
            for col in labels:
                self._data[col] = (
                    self._data[col]
                    .map(self._replace_numpystr)
                    .map(self._to_list)
                    .map(np.array)
                )

    def to_list(self, labels=None):
        if labels is None:
            for col in self._data:
                self._data[col] = self._data[col].map(self._to_list)
        else:
            for col in labels:
                self._data[col] = self._data[col].map(self._to_list)

    def to_dict(self, labels=None):
        if labels is None:
            for col in self._data:
                self._data[col] = self._data[col].map(self._to_dict)
        else:
            for col in labels:
                self._data[col] = self._data[col].map(self._to_dict)

    @staticmethod
    def _to_list(x):
        try:
            return json.loads(x)
        except (TypeError, json.JSONDecodeError):
            return x

    @staticmethod
    def _to_dict(x):
        try:
            x = x.replace("'", '"')
            return json.loads(x)
        except (TypeError, json.JSONDecodeError):
            return x

    @staticmethod
    def _replace_numpystr(x):
        try:
            return re.sub(
                r"\n+",
                "",
                re.sub(
                    r" +",
                    ", ",
                    re.sub(
                        r"\[ +",
                        "[",
                        re.sub(
                            r" +\]",
                            "]",
                            re.sub(r"\.\]", ".0]", re.sub(r"\. ", ".0 ", x)),
                        ),
                    ),
                ),
            )
        except TypeError:
            return x


class _Dir:
    """Directory template"""

    _dir_template = "{save_dir}/benchmark_{benchmark_id}/{experiment_id}/{kind}"

    def __init__(self, experiment_id, benchmark_id, autosave, save_dir):
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

    def make_dirs(self, experiment_id=None, benchmark_id=None):
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

        self._benchmark_dir = f"{self.save_dir}/benchmark_{benchmark_id}"
        self._experiment_dir = f"{self._benchmark_dir}/{experiment_id}"

        d = self._dir_template.format(
            save_dir=self.save_dir,
            benchmark_id=benchmark_id,
            kind=kind,
            experiment_id=experiment_id,
        )
        return d
