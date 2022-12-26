from __future__ import annotations

import datetime, os, pickle, re

from cProfile import run
from typing import Any, Callable, Dict, List, Optional, Union

import dimod
import jijmodeling as jm
import numpy as np
import pandas as pd

from jijbench.components import ID, Artifact, Dir, ExperimentResultDefaultDir, Table
from jijbench.exceptions import StoreResultFailedError
from jijbench.experiment._parser import _parse_dimod_sampleset, _parse_jm_sampleset

np.set_printoptions(threshold=np.inf)


class Experiment:
    """Experiment class
    Manage experimental results as Dataframe and artifact (python objects).
    """

    def __init__(
        self,
        *,
        benchmark_id: Optional[Union[int, str]] = None,
        experiment_id: Optional[Union[int, str]] = None,
        autosave: bool = True,
        save_dir: str = ExperimentResultDefaultDir,
    ):
        """constructor of Experiment class

        Args:
            benchmark_id (Optional[Union[int, str]]): benchmark id for experiment. if None, this id is generated automatically. Defaults to None.
            experiment_id (Optional[Union[int, str]]): experiment id for experiment. if None, this id is generated automatically. Defaults to None.
            autosave (bool, optional): autosave option. if True, the experiment result is stored to `save_dir` directory. Defaults to True.
            save_dir (str, optional): directory for saving experiment results. Defaults to ExperimentResultDefaultDir.
        """
        self.autosave = autosave
        self.save_dir = os.path.normcase(save_dir)

        self._id = ID(
            benchmark_id=benchmark_id,
            experiment_id=experiment_id,
        )
        self._table = Table()
        self._artifact = Artifact()
        self._dir = Dir(
            benchmark_id=self._id.benchmark_id,
            experiment_id=self._id.experiment_id,
            autosave=autosave,
            save_dir=os.path.normcase(save_dir),
        )

        # initialize table index
        self._table.current_index = 0

    @property
    def run_id(self):
        return self._id.run_id

    @property
    def experiment_id(self):
        return self._id.experiment_id

    @property
    def benchmark_id(self):
        return self._id.benchmark_id

    @property
    def table(self):
        return self._table.data

    @property
    def artifact(self):
        return self._artifact.data

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exception_type, exception_value, traceback):
        self.stop()

    def start(self):
        self._id.reset(kind="run")
        self._dir.make_dirs(self.run_id)
        # TODO fix deprecate
        self._table.data.loc[self._table.current_index] = np.nan
        return self

    def stop(self):
        if self.autosave:
            record = self._table.data.loc[self._table.current_index].to_dict()
            self.log_table(record)
            self.log_artifact()

        self._table.save_dtypes(
            os.path.normcase(f"{self._dir.experiment_dir}/dtypes.pkl")
        )
        self._table.current_index += 1

    def store(
        self,
        results: Dict[str, Any],
        *,
        table_keys: Optional[List[str]] = None,
        artifact_keys: Optional[List[str]] = None,
        timestamp: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
    ):
        """store experiment results

        Args:
            results (Dict[str, Any]): ex. `{"num_reads": 10, "results": sampleset}`
            table_keys (list[str], optional): _description_. Defaults to None.
            artifact_keys (list[str], optional): _description_. Defaults to None.
            timestamp: Optional[Union[pd.Timestamp, datetime.datetime]]: timestamp. Defaults to None (current time is recorded).
        """
        try:
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
        except Exception as e:
            msg = f"The solver worked fine, but an error occurred while storing the results (in a format such as a pandas table). -> {e}"
            raise StoreResultFailedError(msg)

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

        ids_data = [self.benchmark_id, self.experiment_id, self.run_id, _timestamp]
        self._table.data.loc[index, ids] = ids_data
        record = self._parse_record(record)
        for key, value in record.items():
            if isinstance(value, (int, float)):
                value_type = type(value)
            elif isinstance(value, Callable):
                value_type = str
                value = re.split(
                    r" at| of", re.split(r"function |method ", str(value))[-1]
                )[0]
            else:
                self._table.data.at[index, key] = object
                value_type = object
            record[key] = value
            self._table.data.at[index, key] = value
            self._table.data[key] = self._table.data[key].astype(value_type)

    def store_as_artifact(
        self,
        record: dict,
        timestamp: Optional[Union[pd.Timestamp, datetime.datetime]] = None,
    ):
        """store as artifact

        Args:
            artifact (_type_): _description_
            timestamp (Optional[Union[pd.Timestamp, datetime.datetime]], optional): _description_. Defaults to None.
        """

        if timestamp is None:
            timestamp = pd.Timestamp.now()
        else:
            timestamp = pd.Timestamp(timestamp)

        self._artifact.timestamp.update({self.run_id: timestamp})
        self._artifact.data.update({self.run_id: record.copy()})

    def _parse_record(self, record):
        """if record includes `dimod.SampleSet` or `jijmodeling.SampleSet`, reconstruct record to a new one.

        Args:
            record (dict): record
        """

        def _update_record():
            if isinstance(new_v, list):
                new_record[new_k] = new_v
            elif isinstance(new_v, np.ndarray):
                new_record[new_k] = new_v
            elif isinstance(new_v, dict):
                new_record[new_k] = new_v
            else:
                if not np.isnan(new_v):
                    new_record[new_k] = new_v

        new_record = {}
        for k, v in record.items():
            if isinstance(v, dimod.SampleSet):
                columns, values = _parse_dimod_sampleset(self, v)
                for new_k, new_v in zip(columns, values):
                    _update_record()
            elif isinstance(v, jm.SampleSet):
                columns, values = _parse_jm_sampleset(self, v)
                for new_k, new_v in zip(columns, values):
                    _update_record()
            else:
                new_record[k] = v
        return new_record

    @classmethod
    def load(
        cls,
        *,
        benchmark_id: Union[int, str],
        experiment_id: Union[int, str],
        autosave: bool = True,
        save_dir: str = ExperimentResultDefaultDir,
    ) -> "Experiment":

        experiment = cls(
            benchmark_id=benchmark_id,
            experiment_id=experiment_id,
            autosave=autosave,
            save_dir=os.path.normcase(save_dir),
        )
        experiment._table = Table.load(
            benchmark_id=benchmark_id,
            experiment_id=experiment_id,
            autosave=autosave,
            save_dir=os.path.normcase(save_dir),
        )
        experiment._artifact = Artifact.load(
            benchmark_id=benchmark_id,
            experiment_id=experiment_id,
            autosave=autosave,
            save_dir=os.path.normcase(save_dir),
        )
        return experiment

    def save(self):
        self._table.save(os.path.normcase(f"{self._dir.table_dir}/table.csv"))
        self._artifact.save(self._dir.artifact_dir)

    def log_table(self, record: dict):
        index = [self._table.current_index]
        df = pd.DataFrame({key: [value] for key, value in record.items()}, index=index)
        file_name = os.path.normcase(f"{self._dir.table_dir}/table.csv")
        df.to_csv(file_name, mode="a", header=not os.path.exists(file_name))

    def log_artifact(self):
        def _is_picklable(obj):
            try:
                pickle.dumps(obj)
                return True
            except TypeError:
                return False

        run_id = self.run_id
        if run_id in self._artifact.data.keys():
            save_dir = os.path.normcase(f"{self._dir.artifact_dir}/{run_id}")

            record = {}
            for key, value in self._artifact.data[run_id].items():
                if _is_picklable(value):
                    if isinstance(value, Callable):
                        value = re.split(
                            r" at| of", re.split(r"function |method ", str(value))[-1]
                        )[0]
                else:
                    value = str(value)
                record[key] = value

            with open(os.path.normcase(f"{save_dir}/artifact.pkl"), "wb") as f:
                pickle.dump(record, f)

            timestamp = self._artifact.timestamp[run_id]
            with open(os.path.normcase(f"{save_dir}/timestamp.txt"), "w") as f:
                f.write(str(timestamp))
