import json
import pathlib
from abc import abstractmethod
from typing import List, Tuple, Dict, Union


class Target:
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def parse(self):
        pass


class JijModelingTarget(Target):
    def __init__(
        self, problem, instance: Union[Tuple[str, Dict], List[Tuple[str, Dict]]]
    ):
        self.problem = problem
        self.instance = instance

    def parse(self):
        if isinstance(self.instance, tuple):
            if isinstance(self.instance[0], str) and isinstance(self.instance[1], dict):
                return [(self.problem, self.instance)]
            else:
                return [(self.problem, instance) for instance in self.instance]
        elif isinstance(self.instance, list):
            return [(self.problem, instance) for instance in self.instance]


class PyQUBOTarget(Target):
    def __init__(self):
        pass

    def parse(self):
        return super().parse()


class QUBOTarget(Target):
    def __init__(self, qubo, bias=None):
        self.qubo = qubo
        self.bias = bias

    def parse(self):
        return super().parse()


class InstanceMixin:
    def _instance_dir(self, size: str):
        instance_dir = (
            pathlib.Path(__file__).parent / "Instances" / size / self.problem_name
        )
        return instance_dir

    def instance_names(self, size: str) -> List[str]:
        instance_dir = self._instance_dir(size)
        instance_name = [p.name.split(".")[0] for p in instance_dir.glob("**/*.json")]
        return instance_name

    def small_instance(self) -> List[Tuple[str, Dict]]:
        return [
            (name, self.load("small", name)) for name in self.instance_names("small")
        ]

    def medium_instance(self) -> List[str]:
        return [
            (name, self.load("medium", name)) for name in self.instance_names("medium")
        ]

    def large_instance(self) -> List[str]:
        return [
            (name, self.load("large", name)) for name in self.instance_names("large")
        ]

    def get_instance(self, size: str, instance_name: str) -> Dict:
        return self.load(size, instance_name=instance_name)

    def load(self, size: str, instance_name: str):
        instance_file_path = None
        for file_path in self._instance_dir(size).glob("**/*.json"):
            if file_path.name.split(".")[0] == instance_name:
                instance_file_path = file_path
        if instance_file_path is None:
            raise FileNotFoundError(
                f"'{size}/{self.problem_name}/{instance_name}.json' is not found."
            )

        instance_path = instance_file_path.resolve()
        with open(instance_path, "r") as f:
            instance_data = json.load(f)

        return instance_data
