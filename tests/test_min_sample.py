import sys

sys.path.append(".")
from jijbench import Benchmark, Experiment, Evaluator
import openjij as oj


def test_min_sample():
    target_problems = ["knapsack"]
    instance_size = "small"
    instance_dir = f"jijbench/Instances/{instance_size}"
    result_dir = f"jijbench/Results/makino/{instance_size}"

    def updater(
        sampler, problem, ph_value, multipliers, optional_args, *args, **kwargs
    ):
        response = sampler(problem, ph_value, multipliers, **optional_args)
        return multipliers, optional_args, response

    # updaterの中で動かすsamplerを定義
    def sampler(problem, ph_value, multipliers, num_reads, num_sweeps, *args, **kwargs):
        sampler = oj.SASampler()
        pyq_obj = problem.to_pyqubo(ph_value, {})
        bqm = pyq_obj.compile().to_bqm(feed_dict=multipliers)

        sampler = oj.SASampler(
            num_reads=num_reads,
            num_sweeps=num_sweeps,
        )
        response = sampler.sample(bqm)
        return response

    optional_args = {"num_reads": 1, "num_sweeps": 100}
    bench = Benchmark(
        updater=updater,
        sampler=sampler,
        target_problems=target_problems,
        optional_args=optional_args,
        instance_dir=instance_dir,
        n_instances_per_problem=1,
        result_dir=result_dir,
    )
    bench.run(max_iters=1)
    # resultsを保存 (ここは明示的に保存させずにrunの内部で自動保存でも良いかもしれない)
    # results.save("directory path")

    # 実験結果をロードできるか確認
    one_result_file = bench.experiments[0].results.result_file
    experiment = Experiment(updater=updater, sampler=sampler, result_dir=result_dir)
    experiment.load(one_result_file)

    # 評価は run の際には行わず（runで自動的に行う場合でも実装は別クラスで)、
    # ロードしたデータを使ってrunとは独立に評価を行えるようにする.
    evaluator = Evaluator(experiment)
    # こんな感じでCallableなオブジェクトを受け取って全結果に対しての評価を行なえるようにしておいてもいいかもしれない。
    evaluator.evaluate()
    evaluator.plot_evaluation_metrics()

    # 保存
    # fevaluator.save("directory path")


if __name__ == "__main__":
    test_min_sample()
