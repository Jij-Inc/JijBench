from fileinput import filename
import json
import pathlib
import glob
from typing import Dict, List


class JijBenchInstance:
    def __init__(self, problem_name: str) -> None:
        self.problem_name = problem_name

    def _instance_dir(self, size: str):
        ins_dir = (
            pathlib.Path(__file__).parent.parent
            / "Instances"
            / size
            / self.problem_name
        )
        return ins_dir.resolve()

    def instance_list(self, size: str) -> List[str]:
        instance_dir = self._instance_dir(size)
        instance_name = [p.name.split(".")[0] for p in instance_dir.glob("**/*.json")]
        return instance_name

    def small_list(self) -> List[str]:
        return self.instance_list("small")

    def medium_list(self) -> List[str]:
        return self.instance_list("medium")

    def large_list(self) -> List[str]:
        return self.instance_list("large")

    def get_instance(self, size: str, instance_name: str) -> Dict:
        return load_instance(
            size, problem_name=self.problem_name, instance_name=instance_name
        )


def load_instance(size: str, problem_name: str, instance_name: str):
    instance_file_path = None
    for file_path in (
        pathlib.Path(__file__).parent.parent / "Instances" / size / problem_name
    ).glob("**/*.json"):
        if file_path.name.split(".")[0] == instance_name:
            instance_file_path = file_path
    if instance_file_path is None:
        raise FileNotFoundError(
            f"'{size}/{problem_name}/{instance_name}.json' is not found."
        )

    instance_path = instance_file_path.resolve()
    with open(instance_path, "r") as f:
        instance_data = json.load(f)

    return instance_data
