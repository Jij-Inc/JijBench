from __future__ import annotations

import json, os, pickle, re

import numpy as np
import pandas as pd

from jijbench.const import Path

__all__ = []


class Table:
    """Table template"""

    id_dtypes = {
        "benchmark_id": object,
        "experiment_id": object,
        "run_id": object,
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
        "num_occurrences": object,
        "num_reads": int,
        "num_sweeps": int,
        "num_feasible": int,
        "num_samples": int,
    }

    violation_dtypes = {
        "{const_name}_violations": object,
        "{const_name}_violation_min": float,
        "{const_name}_violation_mean": float,
        "{const_name}_violation_std": float,
    }

    time_dtypes = {"sampling_time": float, "execution_time": float}

    _dtypes_names = [
        "id_dtypes",
        "energy_dtypes",
        "objective_dtypes",
        "num_dtypes",
        "violation_dtypes",
        "time_dtypes",
    ]

    def __init__(self):
        columns = self.get_default_columns()
        self._data = pd.DataFrame(columns=columns)
        self._current_index = 0

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

    def get_time_columns(self):
        return list(self.time_dtypes.keys())

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
                    .map(self._numpystr_to_liststr)
                    .map(self._to_list)
                    .map(np.array)
                )
        else:
            for col in labels:
                self._data[col] = (
                    self._data[col]
                    .map(self._numpystr_to_liststr)
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

    def save(self, savepath):
        self._data.to_csv(savepath)

    def save_dtypes(self, savepath):
        index = self._data.loc[self._current_index].index
        dtypes = {}
        for i, v in zip(index, self._data.loc[self._current_index]):
            dtypes[i] = type(v)
        with open(os.path.normcase(savepath), "wb") as f:
            pickle.dump(dtypes, f)

    @classmethod
    def load(cls, *, benchmark_id, experiment_id, autosave, save_dir):
        d = Path(
            benchmark_id=benchmark_id,
            experiment_id=experiment_id,
            autosave=autosave,
            save_dir=save_dir,
        )
        table = cls()
        table.data = pd.read_csv(
            os.path.normcase(f"{d.table_dir}/table.csv"), index_col=0
        )
        dtypes = cls.load_dtypes(
            benchmark_id=benchmark_id,
            experiment_id=experiment_id,
            autosave=autosave,
            save_dir=save_dir,
        )
        table.as_finetype(dtypes=dtypes)
        return table

    @classmethod
    def load_dtypes(cls, *, benchmark_id, experiment_id, autosave, save_dir):
        d = Dir(
            benchmark_id=benchmark_id,
            experiment_id=experiment_id,
            autosave=autosave,
            save_dir=save_dir,
        )
        with open(os.path.normcase(f"{d.experiment_dir}/dtypes.pkl"), "rb") as f:
            return pickle.load(f)

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
    def _numpystr_to_liststr(x):
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
