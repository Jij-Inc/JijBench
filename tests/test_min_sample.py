from parasearch import Benchmark, Experiment, Evaluator
import openjij as oj


def test_min_sample():
    target_instances = ["knapsack"]
    instance_size = "small"
    instance_dir = f"parasearch/Instances/{instance_size}"
    result_dir = f"parasaerch/Results/makino/{instance_size}"

    def updater(problem, decoded, params) -> dict:
        return params

    def sampler(problem, ph_value, multipliers, num_reads, num_sweeps):

        sampler = oj.SASampler()

    bench = Benchmark(updater=updater, sampler=sampler)
    results = bench.run()
    # resultsを保存 (ここは明示的に保存させずにrunの内部で自動保存でも良いかもしれない)
    results.save("directory path")

    # Experimentで結果をロードできるようにする
    # このUIにする場合は、Experimentクラスは一度の実験全体の結果を保持しておく必要がある.
    # (全体の結果 == List[現在のExperimentオブジェクト])
    results = Experiment.load("filename")

    # 評価は run の際には行わず（runで自動的に行う場合でも実装は別クラスで)、
    # ロードしたデータを使ってrunとは独立に評価を行えるようにする.
    evaluator = Evaluator(results)
    # こんな感じでCallableなオブジェクトを受け取って全結果に対しての評価を行なえるようにしておいてもいいかもしれない。
    evaluator.evaluate(name="new metric", func=lambda x: ...)

    # 保存
    evaluator.save("directory path")
