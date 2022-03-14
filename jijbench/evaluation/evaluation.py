import jijbench as jb


class Evaluator:
    def __init__(self, experiment: jb.Experiment):
        self.experiment = experiment
    
    @property
    def table(self):
        return self.experiment.table
    
    @property
    def artifact(self):
        return self.experiment.artifact
    
    def aggregate(self):
        pass
    
    

    
