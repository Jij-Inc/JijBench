from __future__ import annotations

<<<<<<< HEAD
import datetime, os, pickle, re

from typing import Any, Callable, Dict, List, Optional, Union

import dimod
import jijmodeling as jm
import numpy as np
import pandas as pd

from jijbench.artifact import Artifact
from jijbench.const import ExperimentResultDefaultDir, Path
from jijbench.id import ID
from jijbench.table import Table
from jijbench.experiment._parser import _parse_dimod_sampleset, _parse_jm_sampleset

np.set_printoptions(threshold=np.inf)


class Experiment:
    """Experiment class
    Manage experimental results as Dataframe and artifact (python objects).
    """

    def __init__(
        self,
        *,
        experiment_id: str | None = None,
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

        self.id = ID(
            experiment=experiment_id,
        )
        self._table = Table()
        self._artifact = Artifact()
        self._dir = Path(
            benchmark_id=self.id.benchmark,
            experiment_id=self.id.experiment,
            autosave=autosave,
            save_dir=os.path.normcase(save_dir),
        )

        # initialize table index
        self._table.current_index = 0

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
        self.id.update(kind="run")
        self._dir.make_dirs(self.id.run)
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

        ids_data = [self.id.benchmark, self.experiment_id, self.run_id, _timestamp]
        self._table.data.loc[index, ids] = ids_data
        record = self._parse_record(record)
        for key, value in record.items():
            if isinstance(value, (int, float)):
                value_type = type(value)
                if isinstance(value, bool):
                    value_type = str
                    value = str(value)
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
=======
import dill
import pandas as pd
import typing as tp
import pathlib

from dataclasses import dataclass
from jijbench.consts.path import DEFAULT_RESULT_DIR
from jijbench.data.mapping import Artifact, Mapping, Table
from jijbench.data.elements.id import ID
from jijbench.data.record import Record


@dataclass
class Experiment(Mapping):
    def __init__(
        self,
        data: tuple[Artifact, Table] | None = None,
        name: str | None = None,
        autosave: bool = True,
        savedir: str | pathlib.Path = DEFAULT_RESULT_DIR,
    ):
        if name is None:
            name = ID().data

        if data is None:
            data = (Artifact(), Table())

        if data[0].name is None:
            data[0].name = name

        if data[1].name is None:
            data[1].name = name

        self.data = data
        self.name = name
        self.autosave = autosave

        if isinstance(savedir, str):
            savedir = pathlib.Path(savedir)
        self.savedir = savedir

    @property
    def artifact(self) -> dict:
        return self.data[0].data

    @property
    def table(self) -> pd.DataFrame:
        t = self.data[1].data
        is_tuple_index = all([isinstance(i, tuple) for i in t.index])
        if is_tuple_index:
            names = t.index.names if len(t.index.names) >= 2 else None
            index = pd.MultiIndex.from_tuples(t.index, names=names)
            t.index = index
        return t

    def __enter__(self) -> Experiment:
        p = self.savedir / str(self.name)
        (p / "table").mkdir(parents=True, exist_ok=True)
        (p / "artifact").mkdir(parents=True, exist_ok=True)
        return self

    def __exit__(self, exception_type, exception_value, traceback) -> None:
        index = (self.name, self.table.index[-1])
        self.table.rename(index={self.table.index[-1]: index}, inplace=True)

        if self.autosave:
            self.save()

    def append(self, record: Record) -> None:
        for d in self.data:
            d.append(record, index_name=("experiment_id", "run_id"))

    def concat(self, experiment: Experiment) -> None:
        from jijbench.functions.concat import Concat

        c = Concat()

        artifact = c([self.data[0], experiment.data[0]])
        table = c([self.data[1], experiment.data[1]])

        self.data = (artifact, table)
        self.operator = c

    def save(self):
        def is_dillable(obj: tp.Any):
            try:
                dill.dumps(obj)
>>>>>>> 642c4cd... fix import
                return True
            except Exception:
                return False

<<<<<<< HEAD
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
=======
        p = self.savedir / str(self.name) / "table" / "table.csv"
        self.table.to_csv(p)

        p = self.savedir / str(self.name) / "artifact" / "artifact.dill"
        record_name = list(self.data[0].operator.inputs[1].data.keys())[0]
        if p.exists():
            with open(p, "rb") as f:
                artifact = dill.load(f)
                artifact[self.name][record_name] = {}
        else:
            artifact = {self.name: {record_name: {}}}

        record = {}
        for k, v in self.artifact[self.name][record_name].items():
            if is_dillable(v):
                record[k] = v
            else:
                record[k] = str(v)
        artifact[self.name][record_name].update(record)

        with open(p, "wb") as f:
            dill.dump(artifact, f)
>>>>>>> 642c4cd... fix import
