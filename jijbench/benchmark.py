import os
import glob
import jijmodeling as jm
import dimod
import json
from abc import ABCMeta, abstractmethod
from typing import Callable, List, Dict, Any
from jijmodeling.transpilers.type_annotations import PH_VALUES_INTERFACE
from jijbench import problems
from jijbench.experiment.experiment import Experiment


class Benchmark:
    def __init__(
        self,
        updater: Callable[[jm.Problem, jm.DecodedSamples, Dict], Dict],
        sampler: Callable[
            [jm.Problem, PH_VALUES_INTERFACE, Dict[str, float], Any], dimod.SampleSet
        ],
        target_problems="all",
        n_instances_per_problem="all",
        instance_files=None,
        optional_args=None,
        instance_dir="./Instances",
        result_dir="./Results",
    ) -> None:
        """create benchmark instance

        Args:
            updater (Callable[[jm.Problem, jm.DecodedSamples, Dict], Dict]): parameter update function
            sampler (Callable[[jm.Problem, PH_VALUES_INTERFACE, Dict[str, float], Any], dimod.SampleSet]): sampler method.
            target_problems (str, optional): target instance name. Defaults to "all".
            n_instances_per_problem (str, optional): number of instance. Defaults to "all".
            optional_args (dict, optional): [description]. Defaults to None.
            instance_dir (str, optional): [description]. Defaults to "./Instances".
            result_dir (str, optional): [description]. Defaults to "./Results".
        """
        self.updater = updater
        self.sampler = sampler
        self.instance_context = _InstanceContext(
            target_problems, n_instances_per_problem, instance_files, instance_dir
        )
        self.result_dir = result_dir
        self.optional_args = optional_args

        self._experiments = []

    @property
    def experiments(self):
        return self._experiments

    @experiments.setter
    def experiments(self, experiments: List[Experiment]):
        self._experiments = experiments

    def setup(self):
        files_by_problem = self.instance_context.files_by_problem()
        for problem_name, instance_files in files_by_problem.items():
            for instance_file in instance_files:
                experiment = Experiment(
                    self.updater,
                    self.sampler,
                    result_dir=self.result_dir,
                    optional_args=self.optional_args,
                )
                experiment.setting.problem_name = problem_name
                experiment.setting.instance_file = instance_file
                self._experiments.append(experiment)
        os.makedirs(self.result_dir, exist_ok=True)

    def run(self, sampling_params={}, max_iters=10, save=True):
        """run experiment

        Args:
            sampling_params (dict, optional): sampler parameters. Defaults to {}.
            max_iters (int, optional): max iteration. Defaults to 10.
        """
        self.setup()
        for experiment in self.experiments:
            instance_file = experiment.setting.instance_file
            print(f">> instance_file = {instance_file}")
            problem = getattr(problems, experiment.setting.problem_name)()
            with open(instance_file, "r") as f:
                ph_value = json.load(f)

            experiment.run(problem, ph_value, max_iters)
            if save:
                savename = instance_file.split("/")[-1].split(".")[0] + ".json"
                experiment.save(savename)


class _InstanceState(metaclass=ABCMeta):
    @abstractmethod
    def files_by_problem(self):
        pass


class _SpecificInstance(_InstanceState):
    def __init__(self, instance_files):
        self.instance_files = instance_files

    def files_by_problem(self):
        return self.instance_files


class _AnyInstance(_InstanceState):
    def __init__(self, target_problems, n_instances, instance_dir):
        if target_problems == "all":
            self.target_problems = problems.__all__
        else:
            self.target_problems = target_problems
        self.n_instances = n_instances
        self.instance_dir = instance_dir

    def files_by_problem(self):
        files = {}
        for name in self.target_problems:
            instance_files = glob.glob(
                f"{self.instance_dir}/{name}/**/*.json", recursive=True
            )
            if isinstance(self.n_instances, int):
                instance_files.sort()
                instance_files = instance_files[: self.n_instances]
            files[name] = instance_files
        return files


class _InstanceContext(_InstanceState):
    def __init__(self, target_problems, n_instances, instance_files, instance_dir):
        if instance_files:
            self.state = _SpecificInstance(instance_files)
        else:
            self.state = _AnyInstance(target_problems, n_instances, instance_dir)

    def files_by_problem(self):
        return self.state.files_by_problem()


if __name__ == "__main__":
    from users.makino.updater import update_simple
    from users.makino.solver import sample_model

    target_problems = ["knapsack"]

    instance_size = "small"

    instance_files = {
        "knapsack": [
            "/home/azureuser/JijBenchmark/jijbench/Instances/small/knapsack/f1_l-d_kp_10_269.json"
        ]
    }
    # instance_files = None
    instance_dir = f"./Instances/{instance_size}"
    result_dir = f"./Results/makino/{instance_size}"

    context = _InstanceContext(target_problems, 2, instance_files, instance_dir)
    files = context.files_by_problem()
    print(files)

    bench = Benchmark(
        update_simple,
        sample_model,
        target_problems=target_problems,
        n_instances_per_problem=1,
        instance_files=instance_files,
        instance_dir=instance_dir,
        result_dir=result_dir,
    )
    # bench.setup()
    # print(bench.experiments)
    bench.run(max_iters=0)
