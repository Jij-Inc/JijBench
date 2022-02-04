from abc import ABCMeta
from typing import Callable, List, Dict, Any
import glob
import jijmodeling as jm
from jijmodeling.transpilers.type_annotations import PH_VALUES_INTERFACE
import dimod
import json
from experiment import Experiment
from jijbench import problems


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

        self._problems = {}
        self._experiments = []

    @property
    def experiments(self):
        return self._experiments

    @experiments.setter
    def experiments(self, experiments: List[Experiment]):
        self._experiments = experiments

    def setup(self):
        """if self.target_instances == "all":
            for name in problems.__all__:
                self.problems[name] = getattr(problems, name)()
        else:
            for name in self.target_instances:
                self.problems[name] = getattr(problems, name)()"""

        for instance_file in self.instance_context.files():
            experiment = Experiment(
                self.updater,
                self.sampler,
                result_dir=self.result_dir,
                optional_args=self.optional_args,
            )
            # experiment.setting.problem_name = name
            # instance_name = instance_file.lstrip(self.instance_dir).rstrip(".json")
            experiment.setting.instance_file = instance_file
            # experiment.setting.mathmatical_model = (
            #    jm.expression.serializable.to_serializable(problem)
            # )
            # with open(instance_file, "rb") as f:
            #    experiment.setting.ph_value = json.load(f)
            #experiment.setting.opt_value = experiment.setting.ph_value.pop(
            #    "opt_value", None
            #)
            self._experiments.append(experiment)

    def run(self, sampling_params={}, max_iters=10):
        """run experiment

        Args:
            sampling_params (dict, optional): sampler parameters. Defaults to {}.
            max_iters (int, optional): max iteration. Defaults to 10.
        """
        self.setup()
        # for experiment in self.experiments:
        #     instance_name = experiment.setting["instance_name"]
        #     print(f">> instance_name = {instance_name}")
        #     experiment.setting["num_iterations"] = max_iters
        #     self.run_for_one_experiment(
        #         experiment=experiment,
        #         max_iters=max_iters,
        #     )
        #     print()


class _InstanceState(metaclass=ABCMeta):
    def files():
        pass


class _SpecificInstance(_InstanceState):
    def __init__(self, instance_files):
        self.instance_files = instance_files

    def files(self):
        return self.instance_files


class _AnyInstance(_InstanceState):
    def __init__(self, target_problems, n_instances, instance_dir):
        if target_problems == "all":
            self.target_problems = problems.__all__
        else:
            self.target_problems = target_problems
        self.n_instances = n_instances
        self.instance_dir = instance_dir

    def files(self):
        for name in self.target_problems:
            instance_files = glob.glob(
                f"{self.instance_dir}/{name}/**/*.json", recursive=True
            )
            if isinstance(self.n_instances, int):
                instance_files.sort()
                instance_files = instance_files[: self.n_instances]
        return instance_files


class _InstanceContext(_InstanceState):
    def __init__(self, target_problems, n_instances, instance_files, instance_dir):
        if instance_files:
            self.state = _SpecificInstance(instance_files)
        else:
            self.state = _AnyInstance(target_problems, n_instances, instance_dir)

    def files(self):
        return self.state.files()


if __name__ == "__main__":
    from users.makino.updater import update_simple
    from users.makino.solver import sample_model

    target_problems = "all"

    instance_size = "small"
    instance_files = [
        "/home/azureuser/JijBenchmark/jijbench/Instances/small/knapsack/f1_l-d_kp_10_269.json"
    ]
    # instance_files = None
    instance_dir = f"./Instances/{instance_size}"
    result_dir = f"./Results/makino/{instance_size}"
    result_dir = "./"

    context = _InstanceContext(target_problems, 5, instance_files, instance_dir)
    files = context.files()
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
    bench.run(max_iters=1)
    fafafaf

    print(bench.experiments)
