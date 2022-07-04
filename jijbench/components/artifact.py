from __future__ import annotations

import os, pickle

import pandas as pd

from jijbench.components.dir import Dir

__all__ = []


class Artifact:
    def __init__(self):
        self._data = {}
        self._timestamp = {}

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def timestamp(self):
        return self._timestamp

    @timestamp.setter
    def timestamp(self, timestamp):
        self._timestamp = timestamp

    @classmethod
    def load(cls, *, benchmark_id, experiment_id, autosave, save_dir):
        d = Dir(
            benchmark_id=benchmark_id,
            experiment_id=experiment_id,
            autosave=autosave,
            save_dir=save_dir,
        )
        artifact = cls()

        dir_names = os.listdir(d.artifact_dir)
        for dn in dir_names:
            load_dir = os.path.normcase(f"{d.artifact_dir}/{dn}")
            if os.path.exists(os.path.normcase(f"{load_dir}/artifact.pkl")):
                with open(os.path.normcase(f"{load_dir}/artifact.pkl"), "rb") as f:
                    artifact.data[dn] = pickle.load(f)
            if os.path.exists(os.path.normcase(f"{load_dir}/timestamp.txt")):
                with open(os.path.normcase(f"{load_dir}/timestamp.txt"), "r") as f:
                    artifact.timestamp[dn] = pd.Timestamp(f.read())
        return artifact

    def save(self, savepath):
        for run_id, v in self._data.items():
            save_dir = os.path.normcase(f"{savepath}/{run_id}")
            with open(os.path.normcase(f"{save_dir}/artifact.pkl"), "wb") as f:
                pickle.dump(v, f)

            timestamp = self._timestamp[run_id]
            with open(os.path.normcase(f"{save_dir}/timestamp.txt"), "w") as f:
                f.write(str(timestamp))
