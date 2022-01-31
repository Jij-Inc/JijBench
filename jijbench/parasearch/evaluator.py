from parasearch.experiment import Experiment


class Evaluator:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

        self.evaluation_metrics = {}

    def evaluate(self):
        pass