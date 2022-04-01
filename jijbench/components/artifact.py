import os
import pickle
import pandas as pd
from .dir import Dir


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
            load_dir = f"{d.artifact_dir}/{dn}"
            if os.path.exists(f"{load_dir}/artifact.pkl"):
                with open(f"{load_dir}/artifact.pkl", "rb") as f:
                    artifact.data[dn] = pickle.load(f)
            if os.path.exists(f"{load_dir}/timestamp.txt"):
                with open(f"{load_dir}/timestamp.txt", "r") as f:
                    artifact.timestamp[dn] = pd.Timestamp(f.read())
        return artifact

    def save(self, savepath):
        for run_id, v in self._data.items():
            save_dir = f"{savepath}/{run_id}"
            with open(f"{save_dir}/artifact.pkl", "wb") as f:
                pickle.dump(v, f)

            timestamp = self._timestamp[run_id]
            with open(f"{save_dir}/timestamp.txt", "w") as f:
                f.write(str(timestamp))