from parasearch.experiment import Experiment



class Evaluator:
    def __init__(self, experiment: Experiment):
        self.experiment = experiment

        instance_name = self.experiment.setting.instance_name

        self.evaluation_metrics = {}

    
    