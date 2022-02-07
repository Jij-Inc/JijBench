import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from jijbench import problems
from jijbench.experiment import Experiment

df = pd.DataFrame(np.zeros((5, 10)))
df.groupby(0).aggregate([max, "var"])


fafafaf
class Evaluator:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment
        self.evaluation_metrics = {}

    def evaluate(
        self,
        num_reads=1,
        num_sweeps_list=[25, 50, 100, 150, 200, 300],
        pr=0.99,
    ):
        setting = self.experiment.setting
        results = self.experiment.results

        evaluation_metrics = {
            "annealing_time": [],
            "feasible_rate_for_baseline": [],
            "feasible_rate_for_new_updater": [],
            "time_to_solution": [],
            "success_probability": [],
            "min_energy": [],
            "mean_eneagy": [],
            "residual_energy": [],
        }
        init_step = "0"
        last_step = str(len(setting.multipliers) - 1)
        init_multipliers = setting.multipliers[init_step]
        updated_multipliers = setting.multipliers[last_step]

        problem = getattr(problems, setting.problem_name)()
        ph_value = setting.ph_value
        optional_args = self.experiment.optional_args

        for num_sweeps in num_sweeps_list:
            baseline = self.experiment.sampler(
                problem,
                ph_value,
                init_multipliers,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                **optional_args,
            )

            baseline_decoded = problem.decode(baseline, ph_value, {})

            response = self.experiment.sampler(
                problem,
                ph_value,
                updated_multipliers,
                num_reads=num_reads,
                num_sweeps=num_sweeps,
                **optional_args,
            )
            decoded = problem.decode(response, ph_value, {})
            tau = response.info["sampling_time"]

            evaluation_metrics["annealing_time"].append(tau)
            evaluation_metrics["feasible_rate_for_baseline"].append(
                len(baseline_decoded.feasibles()) / len(baseline_decoded)
            )
            evaluation_metrics["feasible_rate_for_new_updater"].append(
                len(decoded.feasibles()) / len(decoded)
            )
            if baseline_decoded.feasibles():
                min_energy = baseline_decoded.feasibles().energy.min()
                evaluation_metrics["min_energy"].append(min_energy)
            else:
                evaluation_metrics["min_energy"].append(np.nan)
            
            print(decoded.feasibles())
            if decoded.feasibles():
                energies = decoded.feasibles().energy
                evaluation_metrics["mean_eneagy"].append(energies.mean())
            else:
                evaluation_metrics["mean_eneagy"].append(np.nan)
            
            if baseline_decoded.feasibles() and decoded.feasibles():
                ps = (energies <= min_energy).sum() / len(decoded.solutions) + 1e-16
                evaluation_metrics["time_to_solution"].append(
                    np.log(1 - pr) / np.log(1 - ps) * tau if ps < pr else tau
                )
                evaluation_metrics["success_probability"].append(ps)
                evaluation_metrics["residual_energy"].append(
                    energies.mean() - min_energy
                )
            else:
                evaluation_metrics["time_to_solution"].append(np.nan)
                evaluation_metrics["success_probability"].append(np.nan)
                evaluation_metrics["residual_energy"].append(np.nan)

        self.evaluation_metrics = pd.DataFrame(evaluation_metrics)

        instance_file = self.experiment.setting.instance_file
        instance_name = instance_file.split("/")[-1].split(".")[0]

        self.evaluation_metrics.to_csv(f"{results.table_dir}/{instance_name}.csv")

    def plot_evaluation_metrics(self):
        setting = self.experiment.setting
        results = self.experiment.results

        steps = range(len(setting.multipliers))
        penalties = results.penalties
        best_penalties = [min(value) for value in penalties.values()]

        instance_file = self.experiment.setting.instance_file
        instance_name = instance_file.split("/")[-1].split(".")[0]
        os.makedirs(f"{results.img_dir}/{instance_name}", exist_ok=True)
        if best_penalties:
            plt.plot(steps, best_penalties, marker="o")
            plt.title("step - sum of penalties")
            plt.xlabel("step")
            plt.ylabel("sum of penalties")
            plt.savefig(f"{results.img_dir}/{instance_name}/sum_of_penalties.jpg")

        self.evaluation_metrics.plot(x="annealing_time", y="time_to_solution")
        plt.savefig(f"{results.img_dir}/{instance_name}/time_to_solution.jpg")
        self.evaluation_metrics.plot(x="annealing_time", y="success_probability")
        plt.savefig(f"{results.img_dir}/{instance_name}/success_probability.jpg")
        self.evaluation_metrics.plot(x="annealing_time", y="residual_energy")
        plt.savefig(f"{results.img_dir}/{instance_name}/residual_energy.jpg")


if __name__ == "__main__":
    from users.makino.updater import update_simple
    from users.makino.solver import sample_model

    experiment = Experiment(updater=update_simple, sampler=sample_model)
    experiment.load(
        "/home/azureuser/JijBenchmark/jijbench/Results/makino/small/results_4/logs/f1_l-d_kp_10_269.json"
    )

    evaluator = Evaluator(experiment)
    evaluator.evaluate(num_sweeps_list=[25, 300])
    evaluator.plot_evaluation_metrics()
