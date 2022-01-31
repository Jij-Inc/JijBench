from typing import Callable, List, Dict, Any
import glob
import jijmodeling as jm
from jijmodeling.transpilers.type_annotations import PH_VALUES_INTERFACE
import dimod
import json
from experiment import Experiment
from parasearch import problems


class Benchmark:
    def __init__(
        self,
        updater: Callable[[jm.Problem, jm.DecodedSamples, Dict], Dict],
        sampler: Callable[
            [jm.Problem, PH_VALUES_INTERFACE, Dict[str, float], Any], dimod.SampleSet
        ],
        target_instances="all",
        n_instances_per_problem="all",
        optional_args=None,
        instance_dir="./Instances",
        result_dir="./Results",
    ) -> None:
        """create benchmark instance

        Args:
            updater (Callable[[jm.Problem, jm.DecodedSamples, Dict], Dict]): parameter update function
            sampler (Callable[[jm.Problem, PH_VALUES_INTERFACE, Dict[str, float], Any], dimod.SampleSet]): sampler method.
            target_instances (str, optional): target instance name. Defaults to "all".
            n_instances_per_problem (str, optional): number of instance. Defaults to "all".
            optional_args (dict, optional): [description]. Defaults to {}.
            instance_dir (str, optional): [description]. Defaults to "./Instances".
            result_dir (str, optional): [description]. Defaults to "./Results".
        """
        self.updater = updater
        self.sampler = sampler
        self.target_instances = target_instances
        self.n_instances_per_problem = n_instances_per_problem
        self.instance_dir = instance_dir
        self.result_dir = result_dir
        self.optional_args = optional_args

        self._problems = {}
        self._experiments = []

    @property
    def problems(self):
        return self._problems

    @problems.setter
    def problems(self, problems: Dict[str, jm.Problem]):
        self._problems = problems

    @property
    def experiments(self):
        return self._experiments

    @experiments.setter
    def experiments(self, experiments: List[Experiment]):
        self._experiments = experiments

    def setup(self):
        if self.target_instances == "all":
            for name in problems.__all__:
                self.problems[name] = getattr(problems, name)()
        else:
            for name in self.target_instances:
                self.problems[name] = getattr(problems, name)()

        for name, problem in self.problems.items():
            instance_files = glob.glob(
                f"{self.instance_dir}/{name}/**/*.json", recursive=True
            )
            if isinstance(self.n_instances_per_problem, int):
                instance_files = instance_files[: self.n_instances_per_problem]
            for instance_file in instance_files:
                experiment = Experiment(self.updater, result_dir=self.result_dir)
                experiment.setting.updater = self.updater.__name__
                experiment.setting.problem_name = name
                instance_name = instance_file.lstrip(self.instance_dir).rstrip(".json")
                experiment.setting.instance_name = instance_name
                experiment.setting.mathmatical_model = (
                    jm.expression.serializable.to_serializable(problem)
                )
                with open(instance_file, "rb") as f:
                    experiment.setting.ph_value = json.load(f)
                experiment.setting.opt_value = experiment.setting.ph_value.pop(
                    "opt_value", None
                )
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


if __name__ == "__main__":
    from users.makino.updater import update_simple
    from users.makino.solver import sample_model

    target_instances = ["knapsack"]

    instance_size = "small"
    instance_dir = f"./Instances/{instance_size}"
    result_dir = f"./Results/makino/{instance_size}"

    bench = Benchmark(
        update_simple,
        sample_model,
        target_instances=target_instances,
        n_instances_per_problem=1,
        instance_dir=instance_dir,
        result_dir=result_dir,
    )
    bench.run(max_iters=2)

    print(bench.experiments)
